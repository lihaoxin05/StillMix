import mmcv
from mmcv.runner import EpochBasedRunner


class EpochShowRunner(EpochBasedRunner):
    """OmniSource Epoch-based Runner.

    This runner train models epoch by epoch, the epoch length is defined by the
    dataloader[0], which is the main dataloader.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if 'epoch_runner' in kwargs:
            kwargs.update(epoch=self.epoch)
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        self.outputs = outputs
