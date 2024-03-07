from torch.optim import lr_scheduler

class DelayedReduceOnPlateau(lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose="deprecated", delay = -1):
        
        super().__init__( optimizer=optimizer, mode=mode, factor=0.1, patience=patience,
                 threshold=threshold, threshold_mode=threshold_mode, cooldown=cooldown,
                 min_lr=min_lr, eps=eps, verbose=verbose)

        self.delay = delay

    def _reduce_lr(self, epoch):
        if self.delay<0:
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
        else:
            self.delay -=1
