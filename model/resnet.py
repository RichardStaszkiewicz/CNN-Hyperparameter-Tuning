import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block module for a convolutional neural network.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
    - stride (int): Stride for the convolutional layers.
    - padding (str): Padding mode for the convolutional layers.

    Attributes:
    - conv1 (nn.Conv2d): First convolutional layer.
    - bn1 (nn.BatchNorm2d): Batch normalization layer after the first convolution.
    - conv2 (nn.Conv2d): Second convolutional layer.
    - bn2 (nn.BatchNorm2d): Batch normalization layer after the second convolution.
    - downsample (nn.Module): Downsample layer for adjusting dimensions.
    - relu (nn.ReLU): ReLU activation function.

    Methods:
    - forward(x): Forward pass of the Residual Block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str = "same",
    ) -> None:
        """
        Initializes the Residual Block.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int or tuple[int, int]): Size of the convolutional kernel.
        - stride (int): Stride for the convolutional layers.
        - padding (str): Padding mode for the convolutional layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample: nn.Module = None
        if stride != 0:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Residual Block.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    """
    Residual Neural Network (ResNet) module.

    Parameters:
    - first_conv (dict): Configuration for the first convolutional layer.
    - block_list (list[dict]): List of configurations for Residual Blocks.
    - pool_size (int): Size of the pooling layer.

    Attributes:
    - first_conv (nn.Conv2d): First convolutional layer.
    - res_blocks (nn.ModuleList): List of Residual Blocks.
    - pooling (nn.AvgPool2d): Average pooling layer.

    Methods:
    - forward(x): Forward pass of the ResNet.
    """

    def __init__(self, first_conv: dict, block_list, pool_size: int) -> None:
        """
        Initializes the ResNet.

        Parameters:
        - first_conv (dict): Configuration for the first convolutional layer.
        - block_list (list[dict]): List of configurations for Residual Blocks.
        - pool_size (int): Size of the pooling layer.
        """
        super().__init__()
        self.first_conv = nn.Conv2d(**first_conv)
        self.res_blocks = nn.ModuleList(
            [ResBlock(**block_conf) for block_conf in block_list]
        )
        self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        out = self.first_conv(x)
        for block in self.res_blocks:
            out = block(out)
        out = self.pooling(out)
        return torch.flatten(out, start_dim=1)
