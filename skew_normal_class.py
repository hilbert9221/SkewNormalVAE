'''
Reorganize skew_normal.py into different classes

BUG: No validate_args now.
'''
from torch.distributions import Distribution, register_kl, kl_divergence, Normal, MultivariateNormal
import torch
import math
import numpy as np
import scipy
from functools import partial
import torch.nn.functional as F
from torch import nn


class DiagNormal(Distribution, nn.Module):
    def __init__(self, mu=None, sigma=None, logits=None):
        if mu is not None:
            self.mu = mu
            # sigma means sigma^2 indeed
            self.sigma = sigma
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.mu.shape[:-1], self.mu.shape[-1:], False)

    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., 2 * d]
        '''
        d = logits.size(-1) // 2
        mu = logits[..., :d]
        sigma = logits[..., d:].exp().clamp(min=eps)
        self.mu, self.sigma = mu, sigma

    def weighted_sum(self, weights):
        '''
        Args:
            mu: [..., N, dim]
            weights: [..., K, N]
        '''
        mu, sigma = self.mu, self.sigma
        mu = weights @ mu
        sigma = (weights ** 2) @ sigma
        return mu, sigma

    def sample(self, shape=torch.Size()):
        return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        mu, sigma = self.mu, self.sigma
        dim = int(mu.shape[-1])
        noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
        mu = left_expand(mu, len(shape))
        sigma_sqrt = left_expand(sigma.clamp(min=1e-20).sqrt(), len(shape))
        return mu + noise * sigma_sqrt
    
    def pdf(self, y):
        mu, Sigma = self.mu, torch.diag_embed(self.sigma)
        return normal_pdf(y, mu, Sigma)

    def nll(self,):
        ...

    def log_prob(self, y):
        return - self.nll(y)


class Top1DiagSkewNormal(Distribution):
    def __init__(self, xi=None, omega=None, alpha=None, alpha_prob=None, logits=None):
        if xi is not None:
            self.xi = xi
            self.omega = omega
            self.alpha = alpha
            self.alpha_prob = alpha_prob
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.xi.shape[:-1], self.xi.shape[-1:], False)

    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., 2 * d + d + 1]
        '''
        d = (logits.size(-1) - 1) // 3
        xi = logits[..., :d]
        omega = F.softplus(logits[..., d:2 * d]).clamp(min=eps)
        # omega = logits[..., d:2 * d].exp().clamp(min=eps)
        alpha, alpha_prob = one_of_n_alpha(logits[..., d * 2:], return_prob=True)
        self.xi, self.omega, self.alpha = xi, omega, alpha
        self.alpha_prob = alpha_prob

    def sample_wo_alpha(self, shape=torch.Size()):
        mu, sigma = self.mu, self.omega
        dim = int(mu.shape[-1])
        noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
        mu = left_expand(mu, len(shape))
        sigma_sqrt = left_expand(sigma.clamp(min=1e-20).sqrt(), len(shape))
        return mu + noise * sigma_sqrt

    def sample(self, shape=torch.Size()):
        return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        xi, omega, alpha = self.xi, self.omega, self.alpha
        return sample_canonical_skew_normal(xi, omega, alpha, shape)

    def nll(self, y, add_constant=True, approx_cdf=False, unify=True):
        return nll_skew_normal(y, self.xi, torch.diag_embed(self.omega), self.alpha, add_constant, approx_cdf, unify)

    def log_prob(self, x):
        return - self.nll(x)
    
    def kl(self, p):
        return kl_divergence(self, p)
    

class DiagSkewNormal(Distribution):
    def __init__(self, xi=None, omega=None, alpha=None, logits=None):
        if xi is not None:
            self.xi = xi
            self.omega = omega
            self.alpha = alpha
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.xi.shape[:-1], self.xi.shape[-1:], False)
    
    @property
    def variance(self,):
        return self.cov

    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., 2 * d + d]
        '''
        d = logits.size(-1) // 3
        xi = logits[..., :d]
        omega = F.softplus(logits[..., d:2 * d]).clamp(min=eps)
        # omega = logits[..., d:2 * d].exp().clamp(min=eps)
        alpha = logits[..., 2 * d:]
        self.xi, self.omega, self.alpha = xi, omega, alpha

    def compute_delta(self,):
        delta = (self.alpha / (1 + self.alpha ** 2).sqrt()).sum(-1)
        return delta.mean(), delta.numel()

    def sample_wo_alpha(self, shape=torch.Size()):
        mu, sigma = self.mu, self.omega
        dim = int(mu.shape[-1])
        noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
        mu = left_expand(mu, len(shape))
        sigma_sqrt = left_expand(sigma.clamp(min=1e-20).sqrt(), len(shape))
        return mu + noise * sigma_sqrt

    def sample(self, shape=torch.Size()):
        return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        xi, omega, alpha = self.xi, self.omega, self.alpha
        dim = int(xi.shape[-1])
        scale = 1 / torch.sqrt(1 + (alpha ** 2).sum(-1, keepdim=True))
        delta = (scale * alpha).unsqueeze(-1)
        omega_bar = torch.diag_embed(torch.ones_like(omega))
        omega_aug = torch.cat([delta, omega_bar], dim=-1)
        delta_aug = torch.cat([torch.ones(*xi.shape[:-1], 1, 1, device=xi.device), delta.transpose(-1, -2).contiguous()], dim=-1)
        omega_aug = torch.cat([delta_aug, omega_aug], dim=-2)
        noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
        # omega_sqrt = torch.linalg.cholesky(omega_aug)
        omega_sqrt = cholesky_with_exception(omega_aug)
        noise = (omega_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
        x0 = noise[..., 0]
        x = noise[..., 1:]
        z = torch.zeros_like(x)
        indicator = x0 > 0
        z[indicator] = x[indicator]
        z[~indicator] = - x[~indicator]
        xi = left_expand(xi, len(shape))
        omega_sqrt = left_expand(omega_sqrt, len(shape))
        scale = omega.clamp(min=1e-20).sqrt()
        scale = left_expand(scale, len(shape))
        sample = xi + z * scale
        # t1 = time.time()
        # print(t1 - t0)
        return sample

    def nll(self, y, add_constant=True, approx_cdf=False, unify=False):
        return nll_skew_normal(y, self.xi, torch.diag_embed(self.omega), self.alpha, add_constant, approx_cdf, unify)

    def log_prob(self, x):
        return - self.nll(x)
    
    def kl(self, p):
        return kl_divergence(self, p)


def normal_pdf(x, mu, Sigma):
    '''
    Args:
        x: [..., dim]
        mu: [..., dim]
        Sigma: [..., dim, dim]
    '''
    delta = x - mu
    dim = Sigma.size()[-1]
    inv_Sigma = torch.linalg.inv(Sigma)
    det_Sigma = torch.linalg.det(Sigma)
    quadratic_form = compute_quadratic_form(delta, inv_Sigma, delta)
    constant = (2 * math.pi) ** dim
    pdf = (- quadratic_form / 2).exp() / (det_Sigma * constant).sqrt()
    return pdf


def compute_quadratic_form(x, mat, y):
    '''
    Args:
        x: [..., dim]
        mat: [..., dim, dim]
        y: [..., dim]
    '''
    return (x.unsqueeze(-2) @ mat @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def left_expand(x, n):
    for _ in range(n):
        x = x.unsqueeze(0)
    return x


def batch_diag(x):
    '''
    Args:
        x: [..., dim, dim]
    '''
    n = int(x.shape[-1])
    diag = torch.stack([
        x[..., i, i] for i in range(n)
    ], dim=-1)
    return diag


def get_Omega_bar(Omega):
    Omega_sqrt = batch_diag(Omega).clamp(min=1e-20).sqrt()
    inv_Omega_sqrt = 1. / Omega_sqrt
    inv_Omega_sqrt = torch.diag_embed(inv_Omega_sqrt)
    Omega_bar = inv_Omega_sqrt @ Omega @ inv_Omega_sqrt
    return Omega_bar


def nll_skew_normal(y, xi, Omega, alpha, add_constant=True, approx_cdf=False, unify=True):
        eps = 1e-20
        delta = y - xi
        omega = batch_diag(Omega).clamp(min=eps).sqrt()
        eta = alpha / omega.clamp(min=eps)
        Omega_bar = get_Omega_bar(Omega)
        det_Omega_bar = torch.linalg.det(Omega_bar)
        inv_Omega = torch.linalg.inv(Omega)
        quadratic_form = 1 / 2 * compute_quadratic_form(delta, inv_Omega, delta)
        # log \Phi
        inner_product = (eta * delta).sum(-1)
        if approx_cdf:
            cdf = approx_cdf_normal(inner_product)
        else:
            cdf = cdf_normal(inner_product)
        if unify:
            if add_constant:
                k = int(y.shape[-1])
                pdf = (2 * (- quadratic_form).exp() / ((2 * math.pi) ** (k / 2) * det_Omega_bar.clamp(min=eps).sqrt()) / (omega.prod(-1) * cdf).clamp(min=eps))
                nll = - pdf.clamp(min=eps).log()
            else:
                nll = - ((- quadratic_form).exp() / det_Omega_bar.clamp(min=eps).sqrt() / (omega.prod(-1) * cdf).clamp(min=eps)).clamp(min=eps).log()
        else:
            nll = quadratic_form + det_Omega_bar.clamp(min=eps).log() / 2 + omega.prod(-1).clamp(min=eps).log() - cdf.clamp(min=eps).log()
            if add_constant:
                k = int(y.shape[-1])
                const_pdf = k / 2 * math.log(2 * math.pi)
                const_cdf = - math.log(2)
                const = const_pdf + const_cdf
                nll = nll + const
        return nll


def approx_cdf_normal(x):
    return 1 / 2 * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)).tanh())


def cdf_normal(x):
    return (1 + torch.erf(x / math.sqrt(2))) / 2


def cholesky_with_exception(mat):
    '''
    Args:
        mat: [..., dim, dim]
    '''
    orthogonal, info = torch.linalg.cholesky_ex(mat)
    select = info != 0
    if select.any():
        eps = 1e-4
        dim = mat.size(-1)
        diagonal = eps * torch.eye(dim).to(mat.device)
        nonpositive = mat[select]
        diagonal = left_expand(diagonal, len(mat.shape) - 2)
        nonpositive = nonpositive + diagonal
        intermediate = torch.zeros_like(orthogonal)
        intermediate[~select] = orthogonal[~select]
        intermediate[select] = torch.linalg.cholesky(nonpositive)
        orthogonal = intermediate
    return orthogonal


def fast_cholesky_factorization(y, omega, delta):
    '''
    Args:
        y: [..., dim + 1]
        omega, delta: [..., dim]
    '''
    lower_triangle = (omega - delta ** 2).clamp(min=1e-20).sqrt()
    ret = torch.cat([
        y[..., [0]], lower_triangle * y[..., 1:] + y[..., [0]] * delta
    ], dim=-1)
    return ret


def one_of_n_alpha(logits, positive_scale=False, return_prob=False):
    '''
    Args:
        logits: [..., dim + 1]
    '''
    dim = logits.size()[-1]
    prob_logits, scale_logits = logits[..., :dim - 1], logits[..., [-1]]
    scale = F.softplus(scale_logits) if positive_scale else scale_logits
    # alpha = differentiable_argmax(prob_logits)
    one_hot, index = differentiable_argmax(prob_logits, return_index=True)
    scale_extended = torch.scatter(torch.ones_like(one_hot), -1, index, scale)
    alpha = one_hot * scale_extended
    if return_prob:
        return alpha, prob_logits.softmax(-1)
    else:
        return alpha
    

def differentiable_argmax(logits, dim=-1, return_index=False):
    y_soft = logits.softmax(-1)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    if return_index:
        return ret, index
    else:
        return ret
    

def sample_canonical_skew_normal(xi, omega, alpha, shape=torch.Size(), normalized=True):
    '''
    Args:
        shape: tuple
        xi: [..., dim]
        omega: [..., dim]
        alpha: [..., dim], only one of dim is non-zero

    Return:
        sample: [shape + xi.shape, ..., dim]
    '''
    # create augmented covariance matrix
    dim = int(xi.shape[-1])
    omega_sqrt = omega.clamp(min=1e-20).sqrt()
    # eta = alpha / omega.sqrt()
    omega_bar = torch.ones_like(omega)
    scale = 1 / torch.sqrt(1 + (alpha * omega_bar * alpha).sum(-1, keepdim=True))
    delta = scale * omega_bar * alpha
    noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
    noise = fast_cholesky_factorization(noise, omega_bar, delta)
    x0 = noise[..., 0]
    x = noise[..., 1:]
    z = torch.zeros_like(x)
    indicator = x0 > 0
    z[indicator] = x[indicator]
    z[~indicator] = - x[~indicator]
    xi = left_expand(xi, len(shape))
    omega_sqrt = left_expand(omega_sqrt, len(shape))
    sample = xi + z * omega_sqrt
    return sample
