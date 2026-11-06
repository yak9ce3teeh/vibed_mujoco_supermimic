# Notes

## 11.04

- Vibed an initial version of SuperMimic.
- Aligned the training parameters with those described in the original paper.

## 11.05

- A small 1.1k-round, 3-hour training test on a 10s walking-motion clip.
- Experiment conditions:
  - Original paper: H100, 4096 parallel GPU environments, WM batch size: 2048, policy batch size: 4096.
  - This small test: RTX 4060 Ti, 128 parallel CPU MuJoCo environments, WM batch size: 64, policy batch size: 64.
- The return fluctuated and ended up around 0, with high penalties.
- The policy appears to move toward the target at first, then gradually drifts away.
- Analyzed the accuracy of WM predictions:
  - It looks as though the critic, WM, encoder, and decoder networks have been reduced significantly to fit the current setup. This could be one reason the model is not fitting well.
  - Of course, the training duration is insufficient. I think I'll leave it for now and return to it later.
- Scaled up the current training setup to the same network size as in the original paper, with half as many parallel environments (affordable on a 16GB RTX 4060 Ti).

## 11.06

- Carried out 10-hour training with a paper-sized model on 2,048 parallel environments.
  - It looks like a fantastic result has been achieved, although root position error remains a problem.
  - The return and reward curves appear to be converging well.
  - Foot lifting still has not been achieved, and the robot seems unable to handle movements on the ground.
