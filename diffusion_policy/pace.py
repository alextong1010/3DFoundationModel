import scipy.special as sp
import torch
import matplotlib.pyplot as plt
from scipy.special import erfi as scipy_erfi
import scipy.special as sp
import torch
import matplotlib.pyplot as plt
from scipy.special import erfi as scipy_erfi
import torch

def polyval(x,coeffs):
    """Implementation of the Horner scheme to evaluate a polynomial

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124

    Args:
        x (torch.Tensor): variable
        coeffs (torch.Tensor): coefficients of the polynomial
    """
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x[0]
    return(curVal+coeffs[len(coeffs)-1])


class ERF_1994(torch.nn.Module):
    """Class to compute the error function of a complex number (extends torch.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """
    def __init__(self, n_coefs):
        """Defaul constructor

        Args:
            n_coefs (integer): The number of polynomial coefficients to use in the approximation
        """
        super(ERF_1994, self).__init__()
        # compute polynomial coefficients and other constants
        self.N = n_coefs
        self.i = torch.complex(torch.tensor(0.),torch.tensor(1.))
        self.M = 2*self.N
        self.M2 = 2*self.M
        self.k = torch.linspace(-self.M+1, self.M-1, self.M2-1)
        self.L = torch.sqrt(self.N/torch.sqrt(torch.tensor(2.)))
        self.theta = self.k*torch.pi/self.M
        self.t = self.L*torch.tan(self.theta/2)
        self.f = torch.exp(-self.t**2)*(self.L**2 + self.t**2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real/self.M2
        self.a = torch.flipud(self.a[1:self.N+1])

    def w_algorithm(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L+self.i*z)/(self.L-self.i*z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2*p/(self.L-self.i*z)**2+(1/torch.sqrt(torch.tensor(torch.pi)))/(self.L-self.i*z)
        return w

    def forward(self, z):
        """Compute the error function of a complex number

        The result is computed by manipulating the Faddeeva function.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: erf(z) for each element of z
        """
        # exploit the symmetry of the error function
        # find the sign of the real part
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        # flip sign of imaginary part if negative
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z*self.i)) - z**2) + 1
        return torch.complex(out.real*sign_r, out.imag*sign_i)

    def backward(self, z):
        """Compute the gradient of the error function of a complex number.

        As we know the analytical derivative of the the error function, we can use it directly.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)
        Returns:
            torch.Tensor: grad(erf(z)) for each element of x
        """
        return 2/torch.sqrt(torch.tensor(torch.pi))*torch.exp(-z**2)

erf_torch = ERF_1994(128)

def erfi(x):
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)

    # Convert x to a complex tensor where the real part is zero
    ix = torch.complex(torch.zeros_like(x), x)

    # Compute erf(ix) / i
    erfi_x = erf_torch(ix).imag  # Extract the imaginary part of erf(ix)
    return erfi_x

import logging
from typing import Tuple, Any, Callable, Dict
import torch


def _init_state(
        optimizer: torch.optim.Optimizer,
        p_ref: Dict[torch.Tensor, torch.Tensor],
        s_decay: float,
        betas: Tuple[float],
        s_init: float,
        eps: float,
        store_delta: bool,
        log_every: int,
        force=False):
    '''
    Args:
        optimizer: optimizer instance to initialize extra state for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_every: how often to log scale values.
        force: if True, reinitialize the state.
    '''
    if force or '_pace' not in optimizer.state:
        optimizer.state['_pace'] = {
            's_decay': torch.tensor(s_decay),
            'betas': torch.tensor(betas),
            's_init': torch.tensor(s_init),
            'eps': eps,
            's': torch.zeros(len(betas)),
            'p_ref': {},
            'sum_squared_products': torch.zeros(len(betas)),
            'reward': torch.zeros(len(betas)),
            'max_product': torch.full((len(betas),), 1e-6),
            'sigma': torch.full((len(betas),), 1e-8),
            'iter_count': 0,
            'log_every': log_every,
        }
        _init_reference(optimizer, p_ref, store_delta)

def _init_reference(
        optimizer: torch.optim.Optimizer,
        p_ref: Dict[torch.Tensor, torch.Tensor],
        store_delta: bool):
    '''
    Stores the starting point of the optimization (the "reference").

    Args:
        optimizer: optimizer instance to store reference for.
        p_ref: mapping of parameters to their initial values at the start of optimization.
        store_delta: if true, we should also store the "Delta" value: the
            displacement between the current iterate and the reference.
    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state['_pace'][p] = {
                'ref': p_ref[p].clone(),
            }
            if store_delta:
                optimizer.state['_pace'][p]['delta'] = torch.zeros_like(p)

def _step(
        optimizer: torch.optim.Optimizer,
        base_step: Callable,
        s_decay: float,
        betas: Tuple[float],
        s_init: float,
        eps: float,
        store_delta: bool=True,
        log_every: int=0,
        closure: Callable=None,
        trick1: bool=False,
        trick2: bool=False,
        trick3: bool=False,
        trick3constant = 2.0,
        ):
    '''
    runs one step of pace.

    Args:
        optimizer: pace optimizer instance that we are computing the step for.
        base_step: The "step" function of the base optimizer (e.g. SGD, AdamW etc).
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets between current iterate and reference
            or recompute them on-the-fly.
        force: if True, reinitialize the state.
    Returns:
        loss value
    '''

    prev_grad = torch.is_grad_enabled()

    # we don't wrap the entire function in @torch.no_grad because
    # we want to let the base optimizer differentiate things
    # if it so desires.
    torch.set_grad_enabled(False)


    if closure is not None:
        # if we need to rely on closure to generate gradients
        # then we generate gradient here, but also need to let the
        # base algorithm potentially reevaluate the closure as much
        # as it likes without doubling the gradients the first time it does so.
        # So, we will create a "fake" closure called skip_once_closure
        # to be eventually provided to base_step.
        loss = closure()
        eval_count = 0

        # lie to the base algorithm about first closure eval so that if
        # it thinks that the closure has been evaluated N times at the
        # end of its update, then it will be correct.
        # I'm not sure if this is actually important - might be reasonable
        # to just not do this fake closure stuff.
        def skip_once_closure():
            nonlocal eval_count
            eval_count += 1
            if eval_count == 1:
                return loss
            return closure()
    else:
        skip_once_closure = None

    updates = {}
    grads = {}
    deltas = {}

    global_norm = 0.0
    grad_norm = 0.0

    # store gradients and current parameter values.
    # We need to store the gradients because the base optimizer might
    # change them (for example by adding a weight-decay term).
    # We need to store the current parameter values so that we can
    # compute the "update" generated by the base optimizer by subtracting
    # the "new" values from the current values.
    for group in optimizer.param_groups:
        for p in group['params']:

            if p.grad is None:
                grads[p] = None
            else:
                grads[p] = p.grad.clone()
            updates[p] = p.data.clone()

    # Re-enable gradients and run the base optimizer step
    torch.set_grad_enabled(prev_grad)
    result = base_step(skip_once_closure)
    torch.set_grad_enabled(False)

    # init state after base_step in case base_step only initializes its
    # own state if self.state is empty.
    # Here, we use the fact that updates[p] is the original value of p before the base step
    _init_state(optimizer, updates, s_decay, betas, s_init, eps, store_delta, log_every)
    pace_state = optimizer.state['_pace']


    # compute updates and global norms.
    for group in optimizer.param_groups:
        for p in group['params']:
            if grads[p] is None:
                continue

            p_ref = pace_state[p]['ref']
            if store_delta:
                deltas[p] = pace_state[p]['delta']
            else:
                # Again, we use updates[p] is the original value of p before base_step
                deltas[p] = (updates[p] - p_ref)/(torch.sum(pace_state['s']) + pace_state['eps'])

            updates[p].copy_(p-updates[p])
            p_flat = p.flatten()
            global_norm += torch.dot(p_flat, p_flat)

            g_flat = grads[p].flatten()
            grad_norm += torch.dot(g_flat, g_flat)



    global_norm = torch.sqrt(global_norm)
    grad_norm = torch.sqrt(grad_norm)
    inner_product = 0.0

    # compute inner_product (h in paper pseudocode)
    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            grad = grads[p]

            delta = deltas[p]

            decay = pace_state['s_decay'] * p.flatten() \
                * torch.sum(pace_state['s']) * grad_norm / (global_norm + pace_state['eps'])
            decay = torch.zeros_like(decay) if decay.isnan().any() else decay
            
            product = torch.dot(delta.flatten(), grad.flatten())
            if product.isnan():
                raise ValueError("NaNs in product")
            inner_product += product

            delta.add_(updates[p])

    device = inner_product.device

    for key in pace_state:
        try:
            if pace_state[key].device != device:
                pace_state[key] = pace_state[key].to(device)
        except:
            pass


    # Run the "tuner" step of pace to compute the new s values.
    s = pace_state['s']
    s_decay = pace_state['s_decay']                             # called "lambda" in paper
    s_init = pace_state['s_init']
    betas = pace_state['betas']
    eps = pace_state['eps']
    max_product = pace_state['max_product']                     # called "m" in paper
    reward = pace_state['reward']                               # called "r" in paper
    sum_squared_products = pace_state['sum_squared_products']   # called "v" in paper
    sigma = pace_state['sigma']                                 # called "sigma" in paper
    pace_state['iter_count'] += 1


    max_product.copy_(torch.maximum(
        (betas * max_product), torch.abs(inner_product)))

    sum_squared_products.mul_(
        betas**2).add_(torch.square(inner_product))
    reward.mul_(betas).sub_(s * inner_product)
    reward.copy_(torch.clamp(reward, min=torch.zeros_like(reward)))
    sigma.mul_(betas).sub_(inner_product)
    wealth = max_product * s_init / len(betas) + reward
    f_divisor = erfi(torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0)))
    f_term = s_init / f_divisor
    if trick3:
        constant = trick3constant
    else:
        constant = torch.sqrt(torch.tensor(2.0))
    s_divisor = (constant * torch.sqrt(sum_squared_products) + eps)
    if trick1:
        two_ht_st = 2*max_product*(torch.abs(sigma))
        var_term = torch.sqrt(sum_squared_products + two_ht_st)
        s_divisor = (constant * var_term + eps)
    s_term = erfi(sigma / s_divisor)
    if (f_term * s_term).isnan().any():
        # throw exeception if we get NaNs
        raise ValueError("NaNs in f_term * s_term")
    s.copy_(f_term * s_term)
    # set s = 1
    #s.copy_(torch.ones_like(s)/len(betas))
    #s.copy_(wealth / (torch.sqrt(sum_squared_products) + eps))


    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            p_ref = pace_state[p]['ref']
            delta = deltas[p]
            s_sum = torch.sum(s)
            if trick2:
                mx = max(s_sum, 0.0)
                scale = torch.tensor(1.0)+mx
                p.copy_(p_ref + delta * scale)
            else:
                scale = max(s_sum, 0.0)
                p.copy_(p_ref + delta * scale)



    log_data = {
        'iter_count': pace_state['iter_count'],
        's': torch.sum(s).item(),
    }

    torch.set_grad_enabled(prev_grad)


    return result, log_data


# Empty class used so that we can do isinstance(mechanize(SGD), pace)
class pace:
    pass

def is_pace(opt):
    return isinstance(opt, pace)

def start_pace(
        log_file,
        Base: Any,
        s_decay: float = 0.0,
        betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999,
                               0.99999, 0.999999),
        s_init: float = 1e-8,
        eps: float = 1e-8,
        store_delta: bool = False,
        log_func: Any = None,
        log_every: int = 0,
        trick1: bool=False,
        trick2: bool=False,
        trick3: bool=False,
        trick3constant: float=2,
        ):
    '''
    Wrap a base optimizer class in a pace tuner. The mechanized optimizer
    is a subclass of the base optimizer class in order to minimize disruption
    to subsequent code.

    Args:
        Base: base optimizer class to convert into a pace instance (e.g. torch.optim.SGD)
        s_decay: how much "weight decay" analog to add (called lambda in the paper).
        betas: list of beta values.
        s_init: initial scale value.
        eps: small number for numerical precision.
        store_delta: whether to store the offsets or recompute them on-the-fly.
        log_func: function to call to log data.
            The input to this function will be a dictionary {'iter_count': iteration count, 's': s_value}
            If None, log_func will be set to:
            def log_func(data):
                logger = logging.getLogger(__name__)
                return logger.info(f"(iter={data['iter_count']}), s_sum (global scaling): {data['s']}")
        log_every: how often (in steps) to call log_func.
    '''

    if log_func is None:
        logger = logging.getLogger(__name__)
        log_func = lambda data: logger.info(f"(iter={data['iter_count']}), s_sum (global scaling): {data['s']}")

    class Paced(Base, pace):
        '''
        Wraps a base algorithm as a pace instance.
        '''

        def step(self, closure=None):
            result, log_data = _step(self, super().step, s_decay, betas, s_init, eps, store_delta, log_every, closure, trick1, trick2, trick3, trick3constant)
            # log_dat to log_file
            with open (log_file, 'a') as f:
                f.write(str(log_data) + '\n')
            pace_state = self.state['_pace']
            if log_every > 0 and pace_state['iter_count']%log_every == 0:
                log_func(log_data)
            
            return result

    Paced.__name__ += Base.__name__

    return Paced