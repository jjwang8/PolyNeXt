import functools
import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class BackboneUnfreezeHook(Hook):
    """Freeze backbone forward (no_grad) for the first N iterations,
    then unfreeze so gradients flow normally.

    Works with IterBasedRunner (mmseg segmentation).

    Args:
        unfreeze_iter (int): Iteration at which to unfreeze.
            Default: 1000.
    """

    def __init__(self, unfreeze_iter=1000):
        self.unfreeze_iter = unfreeze_iter
        self.unfrozen = False
        self._original_forward = None

    def before_run(self, runner):
        # Handle both DDP-wrapped and unwrapped models
        if hasattr(runner.model, 'module'):
            backbone = runner.model.module.backbone
        else:
            backbone = runner.model.backbone

        self._original_forward = backbone.forward

        @functools.wraps(self._original_forward)
        def frozen_forward(*args, **kwargs):
            with torch.no_grad():
                return self._original_forward(*args, **kwargs)

        backbone.forward = frozen_forward
        runner.logger.info(
            f'[BackboneUnfreezeHook] Backbone frozen (no_grad) for first {self.unfreeze_iter} iters')

    def before_train_iter(self, runner):
        if runner.iter >= self.unfreeze_iter and not self.unfrozen:
            if hasattr(runner.model, 'module'):
                runner.model.module.backbone.forward = self._original_forward
            else:
                runner.model.backbone.forward = self._original_forward
            self.unfrozen = True
            runner.logger.info(
                f'[BackboneUnfreezeHook] Backbone unfrozen at iter {runner.iter}')