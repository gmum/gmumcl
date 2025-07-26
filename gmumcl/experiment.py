import logging
from copy import deepcopy
from typing import Callable, List, Optional, Dict, Any

from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
import numpy as np
import torch

from gmumcl.util.fabric import setup_fabric
from gmumcl.model import IncrementalClassifier
from gmumcl.method.composer import Composer


def example_train_fn(method, dataloader, task_id):
    """
    Example training function for ExperimentPipeline.
    Args:
        method: Method object with .module and .forward/.backward.
        dataloader: DataLoader for the current task.
        task_id: Integer task index.
    """
    print(f'Training on task {task_id}')
    for X, y, _ in dataloader:
        loss, preds = method.forward(X, y, task_id)
        loss = loss.mean()
        method.backward(loss)

def example_test_fn(method, dataloader, task_id):
    """
    Example testing function for ExperimentPipeline.
    Args:
        method: Method object with .module and .forward.
        dataloader: DataLoader for the current task.
        task_id: Integer task index.
    Returns:
        float: Accuracy in percent.
    """
    print(f'Testing on task {task_id}')
    correct, total = 0, 0
    for X, y, _ in dataloader:
        _, preds = method.forward(X, y, task_id)
        _, preds = torch.max(preds.data, 1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return 100 * correct / total

class ExperimentPipeline:
    """
    Customizable continual learning experiment pipeline with callback support.
    
    Args:
        config (DictConfig): Hydra/OmegaConf config object.
        train_fn (Callable): Training function with signature:
            train_fn(method, dataloader, task_id)
        test_fn (Callable): Testing function with signature:
            test_fn(method, dataloader, task_id)
        callbacks (Optional[List[Callable]]): List of callbacks called after each epoch.

    The pipeline manages scenario, model, method, and dataloaders. It calls train_fn and test_fn for each epoch and task, and collects metrics.

    Example train_fn signature:
        def train_fn(method, dataloader, task_id):
            method.module.train()
            for X, y, _ in dataloader:
                loss, preds = method.forward(X, y, task_id)
                loss = loss.mean()
                method.backward(loss)

    Example test_fn signature:
        def test_fn(method, dataloader, task_id):
            method.module.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X, y, _ in dataloader:
                    _, preds = method.forward(X, y, task_id)
                    _, preds = torch.max(preds.data, 1)
                    total += y.size(0)
                    correct += (preds == y).sum().item()
            return 100 * correct / total

    Example usage:
        from omegaconf import OmegaConf
        from gmumcl.experiment import ExperimentPipeline
        config = OmegaConf.load('path/to/config.yaml')
        pipeline = ExperimentPipeline(config, train_fn, test_fn)
        metrics = pipeline.run()
    """
    def __init__(self, config: DictConfig, train_fn: Callable=example_train_fn, test_fn: Callable=example_test_fn, callbacks: Optional[List[Callable]] = None):
        """
        Initialize the experiment pipeline.
        Args:
            config (DictConfig): Experiment configuration.
            train_fn (Callable): Training function.
            test_fn (Callable): Testing function.
            callbacks (Optional[List[Callable]]): List of callbacks after each epoch.
        Raises:
            ValueError: If train_fn or test_fn is not provided.
        """
        if train_fn is None or test_fn is None:
            raise ValueError("train_fn and test_fn must be provided.")
        self.config = config
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.callbacks = callbacks or []
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        self.metrics: Dict[str, Any] = {}

    def get_scenarios(self):
        """
        Instantiate and return train and test scenarios from config.
        Returns:
            Tuple: (train_scenario, test_scenario)
        """
        dataset_partial = instantiate(self.config.dataset)
        train_dataset = dataset_partial(train=True)
        test_dataset = dataset_partial(train=False)
        scenario_partial = instantiate(self.config.scenario)
        train_scenario = scenario_partial(train_dataset)
        test_scenario = scenario_partial(test_dataset)
        return train_scenario, test_scenario

    def run(self):
        """
        Run the continual learning experiment pipeline.
        Returns:
            dict: Dictionary of collected metrics (accuracy matrix, fwt, bwt, etc).
        """
        config = self.config
        if hasattr(config.exp, 'detect_anomaly') and config.exp.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        calc_bwt = getattr(config.exp, 'calc_bwt', False)
        calc_fwt = getattr(config.exp, 'calc_fwt', False)
        acc_table = getattr(config.exp, 'acc_table', False)
        stop_task = getattr(config.exp, 'stop_after_task', None)
        save_model = hasattr(config.exp, 'model_path')
        model_path = getattr(config.exp, 'model_path', None)

        self.log.info(f'Initializing scenarios')
        train_scenario, test_scenario = self.get_scenarios()

        self.log.info(f'Launching Fabric')
        fabric = setup_fabric(config)

        self.log.info(f'Building model')
        model = fabric.setup(instantiate(config.model))

        self.log.info(f'Setting up method')
        method = instantiate(config.method)(model)

        self.log.info(f'Setting up dataloaders')
        train_tasks = []
        test_tasks = []
        for train_task, test_task in zip(train_scenario, test_scenario):
            train_tasks.append(fabric.setup_dataloaders(DataLoader(
                train_task, 
                batch_size=config.exp.batch_size, 
                shuffle=True, 
                generator=torch.Generator(device=fabric.device)
            )))
            test_tasks.append(fabric.setup_dataloaders(DataLoader(
                test_task, 
                batch_size=1, 
                shuffle=False, 
                generator=torch.Generator(device=fabric.device)
            )))

        N = len(train_scenario)
        R = np.zeros((N, N))
        if calc_fwt:
            b = np.zeros(N)
        for task_id, (train_task, test_task) in enumerate(zip(train_tasks, test_tasks)):
            self.log.info(f'Task {task_id + 1}/{N}')
            if IncrementalClassifier is not None and hasattr(method.module, 'head') and isinstance(method.module.head, IncrementalClassifier):
                self.log.info(f'Incrementing model head')
                method.module.head.increment(train_task.dataset.get_classes())

            self.log.info(f'Setting up task')
            method.setup_task(task_id)

            with fabric.init_tensor():
                for epoch in range(config.exp.epochs):
                    lastepoch = (epoch == config.exp.epochs-1)
                    self.log.info(f'Epoch {epoch + 1}/{config.exp.epochs}')
                    method.module.train()
                    self.train_fn(method, train_task, task_id)
                    method.module.eval()
                    with torch.no_grad():
                        acc = self.test_fn(method, test_task, task_id)
                    if calc_fwt:
                        method_tmp = Composer(
                            deepcopy(method.module), 
                            config.method.criterion, 
                            method.first_lr,
                            method.lr,
                            method.criterion_scale,
                            method.reg_type,
                            method.gamma,
                            method.clipgrad,
                            method.retaingraph,
                            method.log_reg
                        )
                        self.log.info('FWT training pass')
                        method_tmp.setup_task(task_id)
                        method_tmp.module.train()
                        self.train_fn(method_tmp, train_task, task_id)
                        method_tmp.module.eval()
                        with torch.no_grad():
                            b[task_id] = self.test_fn(method_tmp, test_task, task_id)
                    if lastepoch and acc is not None:
                        R[task_id, task_id] = acc
                    if task_id > 0:
                        for j in range(task_id-1, -1, -1):
                            method.module.eval()
                            with torch.no_grad():
                                acc = self.test_fn(method, test_tasks[j], j)
                            if lastepoch:
                                R[task_id, j] = acc
                    for cb in self.callbacks:
                        cb(self, locals())
            if stop_task is not None and task_id == stop_task:
                break
        self.metrics['R'] = R
        if calc_fwt:
            self.metrics['fwt'] = b
        if calc_bwt:
            self.metrics['bwt'] = (R[task_id, :task_id]-R.diagonal()[:-1]).mean() if task_id > 0 else 0
        if calc_fwt:
            fwt = []
            for i in range(1, task_id+1):
                fwt.append(R[i-1, i]-b[i])
            self.metrics['fwt_mean'] = np.array(fwt).mean() if fwt else 0
        if save_model:
            self.log.info(f'Saving model')
            torch.save(model.state_dict(), model_path)
        if acc_table:
            self.log.info(f'Logging accuracy table')
            self.metrics['acc_table'] = R.tolist() if R.size else []
        return self.metrics