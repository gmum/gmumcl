import logging
from typing import Optional, List
from copy import deepcopy

import torch
from torch import nn
from torch import optim


from gmumcl.method.regularization import regularization
from gmumcl.method.method_plugin_abc import MethodPluginABC
from gmumcl.model.cl_module_abc import CLModuleABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Composer:
    """
    Composer class for managing the training process of a module with optional plugins and regularization.

    Attributes:
        module (nn.Module): The module to be trained.
        criterion (nn.Module): The loss function (any torch loss function).
        optimizer (Optional[optim.Optimizer]): The optimizer for training.
        first_lr (float): The learning rate for the first task.
        lr (float): The learning rate for subsequent tasks.
        criterion_scale (float): Scaling factor for the criterion loss when training on subsequent tasks.
        reg_type (Optional[str]): The type of regularization to apply (e.g., L1, L2).
        gamma (Optional[float]): The regularization strength.
        task_heads (bool): Whether to use task-specific heads for multi-task learning.
        clipgrad (Optional[float]): The gradient clipping value.
        retaingraph (Optional[bool]): Whether to retain the computation graph during backpropagation.
        plugins (Optional[list[MethodPluginABC]]): List of plugins to be used during training.
        heads (list[nn.Module]): List of task-specific heads, initialized if `task_heads` is True.
    """

    def __init__(self, 
        module: CLModuleABC,
        criterion: nn.Module,
        first_lr: float, 
        lr: float,
        criterion_scale: float,
        reg_type: Optional[str]=None,
        gamma: Optional[float]=None,
        task_heads: bool=False,
        clipgrad: Optional[float]=None,
        retaingraph: Optional[bool]=False,
        plugins: Optional[List[MethodPluginABC]]=[]
    ):

        """
        Initialize the Composer class.

        Args:
            module (nn.Module): The continual learning module to be trained.
            criterion (nn.Module): The loss function (any torch loss function).
            first_lr (float): The learning rate for the first task.
            lr (float): The learning rate for subsequent tasks.
            criterion_scale (float): Scaling factor for the criterion loss when training on subsequent tasks.
            reg_type (Optional[str], optional): The type of regularization to apply (e.g., L1, L2). Defaults to None.
            gamma (Optional[float], optional): Regularization strength. Defaults to None.
            task_heads (bool, optional): Whether to use task-specific heads for multi-task learning. Defaults to False.
            clipgrad (Optional[float], optional): Maximum gradient norm for gradient clipping. Defaults to None.
            retaingraph (Optional[bool], optional): Whether to retain the computation graph during backpropagation. Defaults to False.
            plugins (Optional[list[MethodPluginABC]], optional): List of method plugins to extend the training process. Defaults to an empty list.
        """

        self.module = module

        self.criterion = criterion
        self.optimizer = None
        self.first_lr = first_lr
        self.lr = lr
        self.criterion_scale = criterion_scale
        self.reg_type = None if (reg_type is None) or (not reg_type) else reg_type
        self.gamma = gamma
        self.task_heads = task_heads
        self.clipgrad = clipgrad
        self.retaingraph = retaingraph
        self.plugins = plugins
        if self.task_heads:
            self.heads = []
        for plugin in self.plugins:
            plugin.set_module(self.module)
            log.info(f'Plugin {plugin.__class__.__name__} added to composer')


    def _setup_optim(self, task_id: int):
        """
        Sets up the optimizer for the model.
        This method initializes the optimizer with the model parameters that require
        gradients. It uses the Adam optimizer with a learning rate that depends on
        the task ID. If the task ID is 0, it uses `first_lr`, otherwise it uses `lr`.

        Args:
            task_id (int): The ID of the current task. Determines the learning rate to use.
        """

        params = list(self.module.parameters())
        params = filter(lambda p: p.requires_grad, params)
        lr = self.first_lr if task_id == 0 else self.lr
        self.optimizer = optim.Adam(params, lr=lr)


    def _add_reg(self, loss):
        """
        Adds regularization to the given loss if gamma and reg_type are set.

        Args:
            loss (float): The original loss value.

        Returns:
            float: The loss value with regularization added if applicable.
        """

        if (self.gamma is not None) and (self.reg_type is not None) and hasattr(self.module, 'activations'):
            loss += self.gamma*regularization(self.module.activations, self.reg_type)
        return loss


    def setup_task(self, task_id: int):
        """
        Set up the task with the given task ID.
        This method initializes the optimizer for the specified task and
        calls the setup_task method on each plugin associated with this instance.

        Args:
            task_id (int): The unique identifier for the task to be set up.
        """

        if self.task_heads:
            if task_id >= len(self.heads):
                tmp_head = self.module.head
                if task_id > 0:
                   tmp_head = deepcopy(self.module.head)
                self.heads.append(tmp_head)
            self.module.head = self.heads[task_id]

        self._setup_optim(task_id)
        for plugin in self.plugins:
            plugin.setup_task(task_id)



    def forward(self, x, y, task_id):
        """
        Perform a forward pass through the model and apply plugins.

        Args:
            x (torch.Tensor): Input tensor to the model.
            y (torch.Tensor): Target tensor for computing the loss.
            task_id (int): The ID of the current task.

        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): The computed loss after applying regularization and plugins.
                - preds (torch.Tensor): The model predictions after applying plugins.
        """

        preds = self.module(x)
        loss = self.criterion(preds, y)
        if task_id > 0:
            loss *= self.criterion_scale
            
        loss_ct = loss

        loss = self._add_reg(loss)
        for plugin in self.plugins:
            loss, preds = plugin.forward(x, y, loss, preds)
        return loss, preds


    def backward(self, loss):  
        """
        Performs a backward pass and updates the model parameters.

        Args:
            loss (torch.Tensor): The loss tensor from which to compute gradients.
            
        This method performs the following steps:
        1. Resets the gradients of the optimizer.
        2. Computes the gradients of the loss with respect to the model parameters.
        3. Optionally clips the gradients to prevent exploding gradients.
        4. Updates the model parameters using the optimizer.
        """

        self.optimizer.zero_grad()
        loss.backward(retain_graph=self.retaingraph)
        if self.clipgrad is not None:
            nn.utils.clip_grad_norm_(self.module.parameters(), self.clipgrad)
        self.optimizer.step()