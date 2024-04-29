import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
import math
import sys
import functools
from typing import Callable
from scipy.special import comb
from math import exp
import dp_accounting
import ml_collections
import numpy as np
import scipy.stats


class DPSGD(SGD):

    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size,
                 sample_size, lr=0.01):
        super(DPSGD, self).__init__(params, lr=lr)

        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad,
                                    max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)

    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def zero_sample_grad(self):
        super(DPSGD, self).zero_grad()

    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()

                ## Adding noise and aggregating each element of the lot:
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale*self.gradient_norm_bound)).to(device)
                p.grad.data = torch.sum(p.grad.data, dim=0) * self.sample_size / self.lot_size
        super(DPSGD, self).step(*args, **kwargs)


class DPAdam(Adam):

    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size,
                 sample_size, lr=0.01):
        super(DPAdam, self).__init__(params, lr=lr)

        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad, max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)

    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def zero_sample_grad(self):
        super(DPAdam, self).zero_grad()

    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()

                ## Adding noise and aggregating each element of the lot:
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale*self.gradient_norm_bound)).to(device)
                p.grad.data = torch.sum(p.grad.data, dim=0) * self.sample_size / self.lot_size
        super(DPAdam, self).step(*args, **kwargs)

class GCN_DPAdam(Adam):

    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size, lr=0.01):
        super(GCN_DPAdam, self).__init__(params, lr=lr)

        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        # self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad, max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)

    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def zero_sample_grad(self):
        super(GCN_DPAdam, self).zero_grad()

    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()
                p.grad.data = torch.sum(p.grad.data, dim=0)
                ## Adding noise and aggregating each element of the lot:
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale)).to(device)
                p.grad.data = p.grad.data / self.lot_size
        super(GCN_DPAdam, self).step(*args, **kwargs)


class GCN_DPSGD(SGD):
    """
    实现GCN使用DP优化器迭代
    """
    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size, lr=0.01):
        super(GCN_DPSGD, self).__init__(params, lr=lr)
        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        # self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []


    def zero_sample_grad(self):
        super(GCN_DPSGD, self).zero_grad()
    
    def zero_accum_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.accumulated_grads = []

    def per_sample_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    per_sample_grad = p.grad.detach().clone()

                    ## Clipping gradient
                    clip_grad_norm_(per_sample_grad,
                                    max_norm=self.gradient_norm_bound)
                    p.accumulated_grads.append(per_sample_grad)
    
    def step(self, device, *args, **kwargs):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # DP:
                ## Adding noise and aggregating on batch:
                p.grad.data = torch.stack(p.accumulated_grads, dim=0).clone()
                p.grad.data = torch.sum(p.grad.data, dim=0)
                p.grad.data += torch.empty(p.grad.data.shape).normal_(mean=0.0, std=(self.noise_scale)).to(device)
                
                p.grad.data = p.grad.data / self.lot_size
        super(GCN_DPSGD, self).step(*args, **kwargs)

class GCN_DP_AC():
    """
    accountant for DP GCN 

    If the norms of the values are bounded ||v_i|| <= C, the noise_multiplier is defined as s / C.
    """
    def __init__(self,noise_scale:float,Ntr:int,m:int,C:float,r=None,delta=None,K=None,max_terms_per_node=None) -> None:
        if not max_terms_per_node:
            self.max_terms_per_node = (math.pow(K,r+1) - 1) / (K - 1)
        else:
            self.max_terms_per_node = max_terms_per_node
        self.noise_multiplier = noise_scale / C
        self.num_samples = Ntr
        
        if delta:
            self.target_delta = delta
        else:
            self.target_delta = 1 / (10 * self.num_samples)

        self.batch_size = m

    def get_privacy(self,epochs:int) -> float:
        self.num_training_steps = epochs
        return multiterm_dpsgd_privacy_accountant(num_training_steps=self.num_training_steps,
                                                  noise_multiplier=self.noise_multiplier,
                                                  target_delta=self.target_delta,
                                                  num_samples=self.num_samples,
                                                  batch_size=self.batch_size,
                                                  max_terms_per_node=self.max_terms_per_node)

class SGD_DP_AC():
    """
    accountant for DPSGD

    'noise_scale': If the norms of the values are bounded ||v_i|| <= C, the noise_multiplier is defined as s / C.
    sampling_probability: The probability of sampling a single sample every batch. For uniform sampling without replacement, this is (batch_size / num_samples)
    Each record in the dataset is included in the sample independently with probability `sampling_probability`. Then the `DpEvent` `event` is applied to the sample of records.
    """
    def __init__(self,noise_scale:float,clip_bound:float,sample_ratio:float,delta=None) -> None:
        # self.num_training_steps = epochs
        self.noise_multiplier = noise_scale / clip_bound
        if not delta:
            self.target_delta = None
        else:
            self.target_delta = delta
        self.sampling_probability = sample_ratio

    def get_privacy(self,epochs:int) -> float:
        return dpsgd_privacy_accountant(num_training_steps=epochs, 
                                        noise_multiplier=self.noise_multiplier,
                                        target_delta=self.target_delta,
                                        sampling_probability=self.sampling_probability)


def multiterm_dpsgd_privacy_accountant(num_training_steps,
                                       noise_multiplier,
                                       target_delta, num_samples,
                                       batch_size,
                                       max_terms_per_node):
    """Computes epsilon after a given number of training steps with DP-SGD/Adam.

    Accounts for the exact distribution of terms in a minibatch,
    assuming sampling of these without replacement.

    Returns np.inf if the noise multiplier is too small.

    Args:
        num_training_steps: Number of training steps.
        noise_multiplier: Noise multiplier that scales the sensitivity.
        target_delta: Privacy parameter delta to choose epsilon for.
        num_samples: Total number of samples in the dataset.
        batch_size: Size of every batch.
        max_terms_per_node: Maximum number of terms affected by the removal of a
        node.

    Returns:
        Privacy parameter epsilon.
    """
    if noise_multiplier < 1e-20:
        return np.inf

    # Compute distribution of terms.
    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    # Compute unamplified RDPs (that is, with sampling probability = 1).
    orders = np.arange(1, 10, 0.1)[1:]

    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
    unamplified_rdps = accountant._rdp  # pylint: disable=protected-access

    # Compute amplified RDPs for each (order, unamplified RDP) pair.
    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node))
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
            order - 1)
        amplified_rdps.append(amplified_rdp)

    # Verify lower bound.
    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(unamplified_rdps *
                    (batch_size / num_samples)**2 <= amplified_rdps + 1e-6):
        raise ValueError('The lower bound has been violated. Something is wrong.')

    # Account for multiple training steps.
    amplified_rdps_total = amplified_rdps * num_training_steps

    # Convert to epsilon-delta DP.
    return dp_accounting.rdp.compute_epsilon(orders, amplified_rdps_total,
                                            target_delta)[0]


def dpsgd_privacy_accountant(num_training_steps, noise_multiplier,
                             target_delta,
                             sampling_probability):
    """Computes epsilon after a given number of training steps with DP-SGD/Adam.

    Assumes there is only one affected term on removal of a node.
    Returns np.inf if the noise multiplier is too small.

    Args:
        num_training_steps: Number of training steps.
        noise_multiplier: Noise multiplier that scales the sensitivity.
        target_delta: Privacy parameter delta to choose epsilon for.
        sampling_probability: The probability of sampling a single sample every
        batch. For uniform sampling without replacement, this is (batch_size /
        num_samples).

    Returns:
        Privacy parameter epsilon.
    """
    if noise_multiplier < 1e-20:
        return np.inf

    orders = np.arange(1, 200, 0.1)[1:]
    event = dp_accounting.PoissonSampledDpEvent(
        sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier))
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(event, num_training_steps)
    return accountant.get_epsilon(target_delta)


def get_training_privacy_accountant(config,
                                    num_training_nodes,
                                    max_terms_per_node):
    """Returns an accountant that computes DP epsilon for a given number of training steps."""
    if not config.differentially_private_training:
        return lambda num_training_steps: 0

    if config.model == 'mlp':
        return functools.partial(
            dpsgd_privacy_accountant,
            noise_multiplier=config.training_noise_multiplier,
            target_delta=1 / (10 * num_training_nodes),
            sampling_probability=config.batch_size / num_training_nodes)
    if config.model == 'gcn':
        return functools.partial(
            multiterm_dpsgd_privacy_accountant,
            noise_multiplier=config.training_noise_multiplier,
            target_delta=1 / (10 * num_training_nodes),
            num_samples=num_training_nodes,
            batch_size=config.batch_size,
            max_terms_per_node=max_terms_per_node)

    raise ValueError(
        'Could not create privacy accountant for model: {config.model}.')



class Mechanism:
    def __init__(self, eps, input_range,device=None, **kwargs):
        self.eps = eps
        self.alpha, self.beta = input_range
        self.device = device

    def __call__(self, x):
        raise NotImplementedError


class MultiBit(Mechanism):
    def __init__(self, *args, m='best', **kwargs):
        super(MultiBit,self).__init__(*args, **kwargs)
        self.m = m

    def __call__(self, x):
        n, d = x.size()
        if self.m == 'best':
            m = int(max(1, min(d, math.floor(self.eps / 2.18))))
        elif self.m == 'max':
            m = d
        else:
            m = self.m

        # sample features for perturbation
        BigS = torch.rand_like(x).topk(m, dim=1).indices
        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, BigS, True)
        
        del BigS

        # perturb sampled features
        em = math.exp(self.eps / m)
        p = (x - self.alpha) / (self.beta - self.alpha)
        p = (p * (em - 1) + 1) / (em + 1)
        t = torch.bernoulli(p)
        x_star = s * (2 * t - 1)
        del p, t, s

        # unbiase the result
        x_prime = d * (self.beta - self.alpha) / (2 * m)
        x_prime = x_prime * (em + 1) * x_star / (em - 1)
        x_prime = x_prime + (self.alpha + self.beta) / 2
        x_prime = x_prime.to(self.device)
        return x_prime

class Duchi(Mechanism):
    def __init__(self, *args, device, d, **kwargs):
        super(Duchi,self).__init__(*args, **kwargs)
        # self.odd = odd
        self.eps
        self.device = device
        
        if d % 2 != 0:
            # d is odd
            C_d = 2 ** (d - 1) / comb(d - 1, int((d - 1) / 2))
        
        else:
            # otherwise
            C_d = (comb(d, int(d / 2)) / comb(d - 1, int(d / 2))) * .5
            C_d += (2 ** (d - 1)/comb(d - 1, int(d / 2))) 
        
        C_d = round(C_d,10)

        B = ((exp(self.eps) + 1) / (exp(self.eps) - 1) * C_d)
        self.B = round(B,10)

    def __call__(self, x):
        # (batch, x)
        b,d = x.shape
        tensor = [self.duchi_multi_dim_method(tp=x[i]) for i in range(b)]
        return torch.concat(tensor,dim=0)
        

    def duchi_multi_dim_method(self,tp:torch.Tensor):
        """
        Duchi et al.’s Solution for Multidimensional Numeric Data

        :param tp: tp is a d-dimensional tuple (t_i \in [-1, 1]^d)
        :param epsilon: privacy budget param
        """

        v = torch.bernoulli(1 / 2 * tp + 1 / 2)
        # v = [self.generate_binary_random(1 / 2 + 1 / 2 * tp[j], 1, -1) for j in range(d)]
        # t_pos = []
        # t_neg = []
        # for t_star in itertools.product([-B, B], repeat=self.d):
        #     t_star = torch.tensor(t_star,device=self.device,dtype=torch.float)
        #     if (t_star@v).item() >= 0:
        #         t_pos.append(t_star)
        #         del t_star
        #     else:
        #         t_neg.append(t_star)
        #         del t_star

        prob = torch.tensor(exp(self.eps) / (exp(self.eps) + 1))
        
    
        if torch.bernoulli(prob).item() == 1:
            t_pos = self.get_tstar(pos=True,v=v)
            return t_pos 
        else:
            t_neg = self.get_tstar(pos=False,v=v)
            return t_neg
        
    def get_tstar(self,pos,v):
        """
        random get a tuple from [-b,b]^d until it satisfies the wanted signal
        """
        t_star = ((torch.bernoulli((torch.ones_like(v)-.5))-.5)*2*self.B).to(self.device)
        
        if pos:
            
            while (t_star@v).item() < 0:
                del t_star
                t_star = ((torch.bernoulli((torch.ones_like(v)-.5))-.5)*2*self.B).to(self.device)
            return t_star.unsqueeze(0)
        else:
            
            while (t_star@v).item() >= 0:
                del t_star
                t_star = ((torch.bernoulli((torch.ones_like(v)-.5))-.5)*2*self.B).to(self.device)
            return t_star.unsqueeze(0)

if __name__ == '__main__':
    m=887
    ns=1
    test = GCN_DP_AC(noise_scale=ns,
              Ntr=m*500,
              m=m,
              C=1,
              max_terms_per_node=(m))
    print(test.get_privacy(500)*500)