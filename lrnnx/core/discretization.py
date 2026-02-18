"""
Discretization methods for continuous-time systems.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor


def zoh(
    A: Tensor, delta: Tensor, integration_timesteps: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """
    Zero-Order Hold (ZOH) discretization method, used across most models.

    .. math::
        \\bar{A} &= \\exp(\\Delta A) \\\\
        \\bar{\\gamma} &= A^{-1} (\\bar{A} - I)

    Reference: https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3

    Args:
        A (torch.Tensor): The continuous-time state matrix.
        delta (torch.Tensor): The discretization step size.
        integration_timesteps (torch.Tensor, optional): Not used in ZOH discretization. Defaults to None.

    Returns:
        A tuple containing:
            - torch.Tensor: A_bar (torch.Tensor): The discretized system matrix.
            - torch.Tensor: gamma_bar (torch.Tensor): The input normalizer.
    """
    Identity = torch.ones(A.shape[0], device=A.device)
    A_bar = torch.exp(delta * A)
    gamma_bar = (1 / A) * (A_bar - Identity)
    return A_bar, gamma_bar


def bilinear(
    A: Tensor, delta: Tensor, integration_timesteps: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    """
    Bilinear method first used in S4.

    .. math::
        \\bar{A} &= (I + 0.5 \\Delta A)^{-1} (I - 0.5 \\Delta A) \\\\
        \\bar{\\gamma} &= (I + 0.5 \\Delta A)^{-1} \\Delta
    
    Reference: https://hazyresearch.stanford.edu/blog/2022-01-14-s4-3

    Args:
        A (torch.Tensor): Continuous-time system matrix, shape: (N,), i.e., only diagonal elements.
        delta (torch.Tensor): Time step for discretization.
        integration_timesteps (torch.Tensor, optional): Not used in bilinear discretization. Defaults to None.

    Returns:
        A tuple containing:
            - torch.Tensor: A_bar : The discretized system matrix.
            - torch.Tensor: gamma_bar : The input normalizer.
    """
    Identity = torch.ones(A.shape[0], device=A.device)
    A_bar = (1 / (Identity + 0.5 * delta * A)) * (Identity - 0.5 * delta * A)
    gamma_bar = (1 / (Identity + 0.5 * delta * A)) * delta
    return A_bar, gamma_bar


def dirac(
    A: Tensor, delta: Tensor, integration_timesteps: Optional[Tensor] = None
) -> tuple[Tensor, float]:
    """
    Dirac discretization method.

    .. math::
        \\bar{A} &= \\exp(\\Delta A) \\\\
        \\bar{\\gamma} &= 1.0
    
    Reference: https://github.com/Efficient-Scalable-Machine-Learning/event-ssm/blob/main/event_ssm/ssm.py

    Args:
        A (torch.Tensor): Continuous-time system matrix.
        delta (torch.Tensor): Time step for discretization.
        integration_timesteps (torch.Tensor, optional): Not used in dirac discretization. Defaults to None.

    Returns:
        A tuple containing:
            - torch.Tensor: A_bar : The discretized system matrix.
            - float: gamma_bar : The input normalizer (1.0).
    """
    A_bar = torch.exp(delta * A)
    gamma_bar = 1.0
    return A_bar, gamma_bar


def async_(
    A: Tensor, delta: Tensor, integration_timesteps: Optional[Tensor] = None
) -> tuple[Tensor, Tensor]:
    r"""
    Asynchronous discretization method, introduced in https://arxiv.org/abs/2404.18508.
    This helps provide a strong inductive bias for asynchronous event-streams.

    .. math::
        \\bar{A} &= \\exp(\\Delta \\cdot \\text{integration\_timesteps} \\cdot A) \\\\
        \\bar{\\gamma} &= A^{-1} (\\exp(\\Delta A) - I)
    
    This method is only for legacy reasons; it is not possible to use this method (or any other
    discretization with async timesteps) with LTI models.

    Args:
        A (torch.Tensor): Continuous-time system matrix.
        delta (torch.Tensor): Time step for discretization.
        integration_timesteps (torch.Tensor): Timesteps for async discretization, ideally of shape (B, L), i.e., difference in timesteps of events.

    Returns:
        A tuple containing:
            - torch.Tensor: A_bar : The discretized system matrix.
            - torch.Tensor: gamma_bar : The input normalizer.
    """
    assert (
        integration_timesteps is not None
    ), "Integration timesteps must be provided for async discretization."
    Identity = torch.ones(A.shape[0], device=A.device)
    A_bar = torch.exp(delta * integration_timesteps * A)
    gamma_bar = (1 / A) * (A_bar - Identity)

    return A_bar, gamma_bar


def no_discretization(
    A: Tensor, delta: Tensor, integration_timesteps: Optional[Tensor] = None
) -> tuple[Tensor, float]:
    """
    No discretization method, identity operation.

    .. math::
        \\bar{A} &= A \\\\
        \\bar{\\gamma} &= 1.0

    Args:
        A (torch.Tensor): Continuous-time system matrix.
        delta (torch.Tensor): Time step for discretization (unused).
        integration_timesteps (torch.Tensor, optional): Not used in no_discretization. Defaults to None.

    Returns:
        A tuple containing:
            - torch.Tensor: A_bar : Same as A.
            - float: gamma_bar : 1.0, as B_bar = B.
    """
    return A, 1.0


DISCRETIZE_FNS: dict[
    str,
    Callable[
        [Tensor, Tensor, Optional[Tensor]], Tuple[Tensor, Union[Tensor, float]]
    ],
] = {
    "zoh": zoh,
    "bilinear": bilinear,
    "dirac": dirac,
    "async": async_,
    "no_discretization": no_discretization,
}