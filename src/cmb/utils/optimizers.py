import torch
import inspect
from dataclasses import dataclass
from torch.optim import Optimizer as TorchOptimizer


class Optimizer:
    """
    Custom optimizer class with support for gradient clipping.

    Attributes:
    - config: Configuration object containing optimizer configurations.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, parameters):
        config_dict = self.config.to_dict()
        optimizer_name = self._get_optimizer_name(config_dict)
        optimizer_cls = self._get_optimizer_class(optimizer_name)
        valid_args = self._get_valid_args(optimizer_cls)
        optimizer_args = {k: v for k, v in config_dict.items() if k in valid_args}
        self.gradient_clip = config_dict.get("gradient_clip", None)
        optimizer = optimizer_cls(parameters, **optimizer_args)
        optimizer = self._wrap_optimizer_step(optimizer)

        return optimizer

    def _wrap_optimizer_step(self, optimizer):
        original_step = optimizer.step

        def step_with_clipping(closure=None):
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]["params"], self.gradient_clip
                )
            original_step(closure)

        optimizer.step = step_with_clipping
        return optimizer

    def _get_optimizer_name(self, config_dict):
        optimizer_names = [
            cls_name
            for cls_name in dir(torch.optim)
            if isinstance(getattr(torch.optim, cls_name), type)
        ]
        possible_names = set(config_dict.keys()) - set(self._get_all_optimizer_args())

        for key in possible_names:
            value = config_dict[key]
            if isinstance(value, str) and value in optimizer_names:
                return value
            elif key in optimizer_names:
                return key
        raise ValueError(
            "Optimizer name not found in configuration. Please specify a valid optimizer name."
        )

    def _get_optimizer_class(self, optimizer_name):
        if hasattr(torch.optim, optimizer_name):
            return getattr(torch.optim, optimizer_name)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_valid_args(self, optimizer_cls):
        signature = inspect.signature(optimizer_cls.__init__)
        valid_args = [
            p.name
            for p in signature.parameters.values()
            if p.name not in ["self", "params"]
        ]
        return valid_args

    def _get_all_optimizer_args(self):
        all_args = set()
        for attr_name in dir(torch.optim):
            attr = getattr(torch.optim, attr_name)
            if inspect.isclass(attr) and issubclass(attr, TorchOptimizer):
                args = self._get_valid_args(attr)
                all_args.update(args)
        return all_args
