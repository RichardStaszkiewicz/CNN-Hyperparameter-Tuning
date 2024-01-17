from typing import Callable
import torch
import torch.nn as nn


class SLPBlock(nn.Module):
    """
    Single-layer perceptron (SLP) block module.

    Parameters:
    - in_size (int): Input size.
    - out_size (int): Output size.
    - activation_fun (str): Activation function name. Options: "relu", "tanh", "sigmoid", "logsoftmax", "none".
    - batch_norm (bool): Whether to apply batch normalization.
    - dropout (float): Dropout rate.

    Attributes:
    - layer (nn.Linear): Linear layer for the perceptron block.
    - bn (nn.BatchNorm1d): Batch normalization layer if applied, else None.
    - act_fun (callable): Activation function.
    - dropout (nn.Dropout): Dropout layer.

    Methods:
    - forward(x): Forward pass of the SLP block.

    Static Attributes:
    - _activation_fun_dict (dict): Dictionary mapping activation function names to corresponding functions.
    """

    _activation_fun_dict: dict = {
        "relu": nn.functional.relu,
        "tanh": nn.functional.tanh,
        "sigmoid": nn.functional.sigmoid,
        "logsoftmax": nn.functional.log_softmax,
        "none": lambda x: x,
    }

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation_fun: str = "relu",
        batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """
        Initializes the SLPBlock module.

        Parameters:
        - in_size (int): Input size.
        - out_size (int): Output size.
        - activation_fun (str): Activation function name.
        - batch_norm (bool): Whether to apply batch normalization.
        - dropout (float): Dropout rate.
        """
        super().__init__()
        self.layer = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size) if batch_norm is True else None
        assert activation_fun in self._activation_fun_dict.keys()
        self.act_fun = self._activation_fun_dict[activation_fun]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SLP block.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        out = self.layer(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.act_fun(out)
        return self.dropout(out)


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) module.

    Parameters:
    - block_list (list): List of dictionaries containing configuration for SLPBlock.

    Attributes:
    - blocks (nn.ModuleList): List of SLPBlock modules.

    Methods:
    - forward(x): Forward pass of the MLP.
    """

    def __init__(self, block_list: list) -> None:
        """
        Initializes the MLP module.

        Parameters:
        - block_list (list): List of dictionaries containing configuration for SLPBlock.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [SLPBlock(**block_conf) for block_conf in block_list]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out
