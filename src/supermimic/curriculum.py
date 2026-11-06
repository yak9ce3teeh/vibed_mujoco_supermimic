"""One-second start bins, with explicit local conventions for unspecified details."""
import numpy as np


class StartBins:
    def __init__(self,motions):
        self.names=list(motions)
        self.bins=np.asarray([(record,start,min(start+50,len(motions[name].states)-1)) for record,name in enumerate(self.names) for start in range(0,len(motions[name].states)-1,50)],dtype=np.int64)
        n=len(self.bins);self.sums=np.zeros(n);self.counts=np.zeros(n,dtype=np.int64)
        self.historic=np.ones(n)/n;self.historic_count=1

    def probabilities(self,update_history=True):
        known=self.counts>0
        fallback=np.mean(self.sums[known]/self.counts[known]) if known.any() else 0.
        mean=np.divide(self.sums,self.counts,out=np.full_like(self.sums,fallback),where=known)
        logits=-25*mean/10.;soft=np.exp(logits-logits.max());soft/=soft.sum()
        pmf=.5/len(soft)+.4*soft+.1*self.historic
        if update_history:
            self.historic=(self.historic*self.historic_count+soft)/(self.historic_count+1)
            self.historic_count+=1
        return pmf/pmf.sum()

    def update(self,bin_ids,discounted_returns):
        np.add.at(self.sums,np.asarray(bin_ids,dtype=int),discounted_returns)
        np.add.at(self.counts,np.asarray(bin_ids,dtype=int),1)

    def state_dict(self):
        return dict(names=self.names,bins=self.bins.copy(),sums=self.sums.copy(),counts=self.counts.copy(),historic=self.historic.copy(),historic_count=self.historic_count)

    def load_state_dict(self,state):
        if state['names']!=self.names or not np.array_equal(state['bins'],self.bins):raise ValueError('Curriculum recording/bin identity differs')
        self.sums=state['sums'].copy();self.counts=state['counts'].copy();self.historic=state['historic'].copy();self.historic_count=state['historic_count']
