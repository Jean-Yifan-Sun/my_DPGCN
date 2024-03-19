import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.nn.utils.clip_grad import clip_grad_norm_
import math
import sys
import functools
from typing import Callable

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

class GCN_DPSGD(SGD):
    """
    实现GCN使用DP优化器迭代
    """
    def __init__(self, params, noise_scale, gradient_norm_bound, lot_size,
                 sample_size, lr=0.01):
        super(GCN_DPSGD, self).__init__(params, lr=lr)
        self.noise_scale = noise_scale
        self.gradient_norm_bound = gradient_norm_bound
        self.lot_size = lot_size
        self.sample_size = sample_size
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.accumulated_grads = []


    def zero_sample_grad(self):
        super(GCN_DPSGD, self).zero_grad()

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
        super(DPSGD, self).step(*args, **kwargs)

class GCN_DP_AC():
    """
    accountant for DP GCN 

    If the norms of the values are bounded ||v_i|| <= C, the noise_multiplier is defined as s / C.
    """
    def __init__(self,noise_scale:float,K:int,Ntr:int,m:int,r:int,C:float,epochs:int,delta:None) -> None:
        self.max_terms_per_node = (K ^ (r - 1) - 1) / (K - 1)
        self.num_training_steps = epochs
        self.noise_multiplier = noise_scale / C
        self.num_samples = Ntr
        
        if delta:
            self.target_delta = delta
        else:
            self.target_delta = 1 / (10 * self.num_samples)

        self.batch_size = m

    def get_privacy(self) -> tuple:
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
    def __init__(self,noise_scale:float,clip_bound:float,sample_ratio:float,epochs:int,delta:None) -> None:
        self.num_training_steps = epochs
        self.noise_multiplier = noise_scale / clip_bound
        self.target_delta = delta
        self.sampling_probability = sample_ratio

    def get_privacy(self) -> float:
        return dpsgd_privacy_accountant(num_training_steps=self.num_training_steps, 
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
