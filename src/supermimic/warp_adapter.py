"""Stock MuJoCo Warp G1 batch physics with the native name/PD contract.

No skate-sim controllers or ordering monkeypatches. Physics is not autograd;
policy differentiation remains inside the learned world model.
"""
from contextlib import contextmanager
import importlib.metadata
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw


@wp.kernel
def _position_pd(
    qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float), actuator_joint: wp.array(dtype=int),
    qpos_index: wp.array(dtype=int), qvel_index: wp.array(dtype=int),
    kp: wp.array(dtype=float), kd: wp.array(dtype=float),
    low: wp.array(dtype=float), high: wp.array(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    world, actuator = wp.tid()
    joint = actuator_joint[actuator]
    torque = float(0.0)
    if joint >= 0:
        torque = kp[joint] * (target[world, joint] - qpos[world, qpos_index[joint]])
        torque = torque - kd[joint] * qvel[world, qvel_index[joint]]
        torque = wp.clamp(torque, low[joint], high[joint])
    ctrl[world, actuator] = torque


class MujocoWarpAdapter:
    """N independent lanes, four 5ms PD substeps per 50Hz control action."""
    def __init__(self, native, num_envs, device='cuda:0', *, nconmax=128, njmax=512,
                 record_torques=False, check_every_step=True):
        if not torch.cuda.is_available():
            raise RuntimeError('Warp physics requires a visible CUDA GPU; no CPU fallback')
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('Warp physics requires --device cuda')
        if self.device.index is None:self.device=torch.device('cuda',torch.cuda.current_device())
        if not isinstance(num_envs,int) or num_envs<1:raise ValueError('num_envs must be positive')
        if min(nconmax,njmax)<1:raise ValueError('Nonzero contact/constraint capacities required')
        versions={name:importlib.metadata.version(name) for name in ['mujoco','mujoco-warp','warp-lang']}
        if versions!={'mujoco':'3.11.0','mujoco-warp':'3.11.0','warp-lang':'1.17.0'}:
            raise RuntimeError(f'Unverified Warp dependency versions: {versions}; use pinned warp environment')
        self.num_envs=num_envs;self.model=native.model
        if self.model.nq!=36 or self.model.nv!=35 or len(native.joint_names)!=29:
            raise ValueError('Requires native 29DoF G1 qpos36/qvel35 model')
        self.control_dt=native.control_dt;self.decimation=native.decimation
        if self.decimation!=4 or self.control_dt!=.02 or self.model.opt.timestep!=.005:
            raise ValueError('Warp adapter requires the explicit four 5ms clock')
        self.joint_names=native.joint_names
        self.qpos_indices=torch.as_tensor(native.qpos_indices,device=self.device,dtype=torch.long)
        self.qvel_indices=torch.as_tensor(native.qvel_indices,device=self.device,dtype=torch.long)
        self.actuator_indices=torch.as_tensor(native.actuator_indices,device=self.device,dtype=torch.long)
        self.q_default=torch.as_tensor(native.q_default,device=self.device,dtype=torch.float32)
        self.action_scale=torch.as_tensor(native.action_scale,device=self.device,dtype=torch.float32)
        wp.init()
        self.stream=wp.stream_from_torch(torch.cuda.current_stream(self.device))
        self.check_every_step=check_every_step;self.record_torques=record_torques
        with self.scope():
            self.warp_model=mjw.put_model(self.model)
            self.data=mjw.make_data(self.model,nworld=num_envs,nconmax=nconmax,njmax=njmax)
        self.qpos=wp.to_torch(self.data.qpos);self.qvel=wp.to_torch(self.data.qvel)
        self.ctrl=wp.to_torch(self.data.ctrl);self.time=wp.to_torch(self.data.time)
        self.overflow=wp.to_torch(self.data.overflow)
        self.steps=torch.zeros(num_envs,device=self.device,dtype=torch.long)
        self.last_action=torch.zeros((num_envs,29),device=self.device)
        self.target=torch.empty((num_envs,29),device=self.device)
        self.target_wp=wp.from_torch(self.target,dtype=wp.float32)
        actuator_joint=np.full(self.model.nu,-1,dtype=np.int32)
        actuator_joint[native.actuator_indices]=np.arange(29,dtype=np.int32)
        with self.scope():
            self.actuator_joint=wp.array(actuator_joint,dtype=wp.int32)
            self.qpos_index=wp.array(native.qpos_indices.astype(np.int32),dtype=wp.int32)
            self.qvel_index=wp.array(native.qvel_indices.astype(np.int32),dtype=wp.int32)
            self.kp=wp.array(native.kp.astype(np.float32),dtype=wp.float32)
            self.kd=wp.array(native.kd.astype(np.float32),dtype=wp.float32)
            self.low=wp.array(native.torque_range[:,0].astype(np.float32),dtype=wp.float32)
            self.high=wp.array(native.torque_range[:,1].astype(np.float32),dtype=wp.float32)
        self.torque_trace=torch.empty((4,num_envs,29),device=self.device) if record_torques else None
        self.metadata=dict(backend='warp',implementation='stock mujoco_warp, no ordering patches',
            num_envs=num_envs,device=str(self.device),versions=versions,physics_dtype='float32',
            physics_dt=.005,decimation=4,control_dt=.02,nconmax=nconmax,njmax=njmax,
            native_joint_order=list(native.joint_names),actuator_indices=native.actuator_indices.tolist(),
            qpos_indices=native.qpos_indices.tolist(),qvel_indices=native.qvel_indices.tolist(),
            root_actuators=sorted(set(range(self.model.nu))-set(native.actuator_indices.tolist())),
            canonical_eval_backend='native',partial_reset=True,
            collision_limitations=['Stock MULTICCD: CYLINDER/CYLINDER and CYLINDER/MESH generate at most one contact; XML is unchanged'],
            equivalence='Same model/controller task; float32 Warp is not claimed bitwise native-equivalent')
        self.reset()

    @contextmanager
    def scope(self):
        with wp.ScopedDevice(str(self.device)), wp.ScopedStream(self.stream):yield

    def _batch(self, value, width):
        result=torch.as_tensor(value,device=self.device,dtype=torch.float32)
        if result.shape==(width,):result=result.expand(self.num_envs,-1)
        if result.shape!=(self.num_envs,width):raise ValueError(f'Expected ({self.num_envs},{width}) state')
        if not torch.isfinite(result).all():raise ValueError('Nonfinite reset state')
        return result

    @torch.no_grad()
    def reset(self,qpos=None,qvel=None,mask=None):
        q=self._batch(self.model.qpos0 if qpos is None else qpos,36)
        v=self._batch(np.zeros(35) if qvel is None else qvel,35)
        if not torch.allclose(torch.linalg.vector_norm(q[:,3:7],dim=-1),torch.ones(self.num_envs,device=self.device),atol=1e-5):
            raise ValueError('Reset requires unit WXYZ root quaternions')
        mask=torch.ones(self.num_envs,device=self.device,dtype=torch.bool) if mask is None else torch.as_tensor(mask,device=self.device,dtype=torch.bool)
        if mask.shape!=(self.num_envs,):raise ValueError('Reset mask must contain one bool per lane')
        with self.scope():mjw.reset_data(self.warp_model,self.data,reset=wp.from_torch(mask.contiguous(),dtype=wp.bool))
        self.qpos[mask]=q[mask];self.qvel[mask]=v[mask]
        self.steps[mask]=0;self.last_action[mask]=0
        with self.scope():mjw.forward(self.warp_model,self.data)
        self.check()
        return self.state()

    def state(self):return torch.cat([self.qpos,self.qvel],dim=-1)

    @torch.no_grad()
    def step(self,action):
        action=torch.as_tensor(action,device=self.device,dtype=torch.float32)
        if action.shape!=(self.num_envs,29):raise ValueError('One 29D action per CUDA lane required')
        if not bool(torch.isfinite(action).all()):raise ValueError('Nonfinite action rejected before changing physics')
        self.last_action.copy_(action);self.target.copy_(self.q_default+self.action_scale*action)
        with self.scope():
            for substep in range(4):
                wp.launch(_position_pd,dim=(self.num_envs,self.model.nu),inputs=[self.data.qpos,self.data.qvel,
                    self.target_wp,self.actuator_joint,self.qpos_index,self.qvel_index,self.kp,self.kd,self.low,self.high,self.data.ctrl])
                if self.record_torques:self.torque_trace[substep].copy_(self.ctrl[:,self.actuator_indices])
                mjw.step(self.warp_model,self.data)
            mjw.forward(self.warp_model,self.data)
        self.steps.add_(1)
        if self.check_every_step:self.check()
        return self.state()

    def synchronize(self):
        with self.scope():wp.synchronize_stream(self.stream)

    def check(self):
        if bool((self.overflow!=0).any()):
            raise RuntimeError(f'MuJoCo Warp capacity overflow: {self.overflow.cpu().tolist()}')
        if not bool(torch.isfinite(self.qpos).all() & torch.isfinite(self.qvel).all()):
            raise FloatingPointError('Nonfinite Warp physics state; no fallback')
