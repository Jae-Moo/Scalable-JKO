# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 2. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# 3. https://github.com/nhartland/KL-divergence-estimators (MIT License)
# ---------------------------------------------------------------

import warnings
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

def select_phi(name):
    if name == 'linear':
        def phi(x):
            return x
            
    elif name == 'kl':
        def phi(x):
            return torch.exp(x) - 1
    
    elif name == 'chi':
        def phi(x):
            y = F.relu(x+2)-2
            return 0.25 * y**2 + y
        
    elif name == 'softplus':
        def phi(x):
            return 2*F.softplus(x) - 2*F.softplus(0*x)
    else:
        raise NotImplementedError
    
    return phi

# ------------------------
# Loss scheduler
# ------------------------   
class Loss_inv:
    def __init__(self, loss_name, init_step=0, scheduler_name=None, num_iteration=None, step_size=None, gamma=None, alpha_min=None, alpha_max=1, warmup=0, schedule_until=None):
        self.phi = select_phi(loss_name)
        self.alpha = 1
        self.count_steps = init_step
        if schedule_until is None: schedule_until=num_iteration
        self.scheduler = self.get_scheduler(scheduler_name, num_iteration, step_size, gamma, alpha_min, alpha_max, warmup, schedule_until)


    def __call__(self, x):
        return self.alpha * self.phi(x / self.alpha)


    def step(self):
        if self.scheduler is not None:
            self.count_steps += 1
            self.alpha = self.scheduler(self.count_steps)

    def get_scheduler(self, scheduler_name, num_iteration, step_size, gamma, alpha_min, alpha_max, warmup, schedule_until):
        if scheduler_name is None or scheduler_name=='none':
            scheduler = None
        elif scheduler_name.lower() == 'steplr':
            assert gamma >= 1
            scheduler = self.steplr(step_size, gamma, alpha_min, alpha_max, warmup)
        elif scheduler_name.lower() == 'linear':
            scheduler = self.linear(num_iteration, alpha_min, alpha_max, warmup, schedule_until)
        elif scheduler_name.lower() == 'cosine':
            scheduler = self.cosine(num_iteration, alpha_min, alpha_max, warmup, schedule_until)
        else:
            raise NotImplementedError
        
        return scheduler

    
    def steplr(self, step_size, gamma, alpha_min, alpha_max, warmup):
        return StepLR(step_size, gamma, alpha_min, alpha_max, warmup)

    def linear(self, num_iteration, alpha_min, alpha_max, warmup, schedule_until):
        
        def scheduler(step):
            step_sch, num_iter_sch = max(step-warmup, 0), max(min(num_iteration,schedule_until) -warmup, 0)
            t = min(step_sch / num_iter_sch, 1.)
            return (1-t)*alpha_min + t*alpha_max
        
        return scheduler

    def cosine(self, num_iteration, alpha_min, alpha_max, warmup, schedule_until):
        
        def scheduler(step):
            step_sch, num_iter_sch = max(step-warmup, 0), max(min(num_iteration,schedule_until)-warmup, 0)
            t = torch.tensor(torch.pi * min(step_sch / num_iter_sch, 1.))
            # return alpha_min + 0.5*(alpha_max-alpha_min)*(1+torch.cos(t))
            return alpha_min + 0.5*(alpha_max-alpha_min)*(1-torch.cos(t))


        return scheduler

class StepLR:
    def __init__(self, step_size, gamma, alpha_min, alpha_max, warmup):
        self.step_size = step_size
        self.gamma = gamma
        self.alpha_min = alpha_min
        self.alpha = alpha_max
        self.warmup = warmup
    
    def __call__(self, step):
        step_sch = max(step-self.warmup, 0)
        if step_sch % self.step_size == self.step_size-1:
            self.alpha = max(self.gamma*self.alpha, self.alpha_min)
        return self.alpha
    
# ------------------------
# EMA
# ------------------------
class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        '''
        EMA Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
        '''
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()

