from __future__ import annotations

import math
from typing import Iterable

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization via quintic Newton-Schulz iterations."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = grad.bfloat16()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()

    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        at = x @ x.t()
        bt = b * at + c * (at @ at)
        x = a * x + bt @ x

    if transposed:
        x = x.t()
    return x


class Muon(torch.optim.Optimizer):
    """Moonlight-style Muon with AdamW fallback for non-matrix parameters.

    - Matrix-like parameters (ndim >= 2): SGD-momentum + Newton-Schulz orthogonalized update.
    - Others (bias/norm etc): AdamW update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        muon_scale: float = 0.2,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            muon_scale=muon_scale,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _use_muon(param: torch.Tensor) -> bool:
        return param.ndim >= 2

    @staticmethod
    def _as_matrix(t: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if t.ndim == 2:
            return t, t.shape
        return t.view(t.size(0), -1), t.shape

    @staticmethod
    def _adjusted_lr(base_lr: float, shape: tuple[int, int], muon_scale: float) -> float:
        m, n = shape
        return base_lr * (muon_scale * math.sqrt(max(m, n)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            muon_scale = group["muon_scale"]

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]

                if self._use_muon(p):
                    g_mat, orig_shape = self._as_matrix(grad)
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g_mat)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g_mat)

                    g_eff = g_mat.add(buf, alpha=momentum) if nesterov else buf
                    u = zeropower_via_newtonschulz5(g_eff, steps=ns_steps).to(dtype=p.dtype)
                    step_lr = self._adjusted_lr(lr, u.shape, muon_scale)

                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(u.view(orig_shape), alpha=-step_lr)
                    continue

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-step_size)

        return loss
