from typing import Union, List
from omegaconf import ListConfig

import torch
import torch.nn as nn

from gmumcl.model.cl_module_abc import CLModuleABC
from gmumcl.model.inc_classifier import IncrementalClassifier


class MLP(CLModuleABC):
    """
    Multi-Layer Perceptron (MLP) class that extends CLModuleABC.

    Args:
        initial_out_features (int): The number of output features for the initial layer.
        sizes (list[int]): A list of integers representing the sizes of each layer.
        layers (list[str]): A list of strings representing the type of each layer.
        head_type (str, optional): The type of the head layer. Defaults to "Normal".
        activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
        **kwargs: Additional keyword arguments for layer instantiation.

    Attributes:
        layers (nn.ModuleList): A list of instantiated layers.

    Methods:
        forward(x):
            Performs a forward pass through the network and records activations.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the head layer.
    """

    def __init__(self,
        initial_out_features: int,
        sizes: List[int],
        activation: nn.Module=nn.Tanh(),
        config: Union[dict, List[dict], ListConfig]={},
    ):
        """
        Initializes the MLP model.
        
        Args:
            initial_out_features (int): The initial number of output features.
            sizes (list[int]): A list of integers representing the sizes of each layer.
            activation (nn.Module, optional): The activation function to use between layers. Defaults to nn.Tanh().
        """
        assert len(sizes) >= 2, "sizes must have at least input and output size"

        super().__init__(
            IncrementalClassifier(
                sizes[-1], 
                initial_out_features
            )
        )

        layers = []
        N = len(sizes)-1
        for i in range(N):
            in_size = sizes[i]
            out_size = sizes[i+1]
            layers.append(nn.Linear(in_size, out_size))
            if i < N-1:
                layers.append(activation)
        self.layers = nn.ModuleList(layers)
       
    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.
            
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        self.reset_activations()

        x = torch.flatten(x, start_dim=1)
        for layer in self.layers:
            x = layer(x)
            self.add_activation(layer, x)
        return self.head(x)