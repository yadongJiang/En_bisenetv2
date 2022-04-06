from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, 
                 optimizer, 
                 max_iters, 
                 power=0.9, 
                 last_epoch=-1, 
                 min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

class WarmupLrScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer, 
                 warmup_iter=1000, 
                 warmup_ratio=5e-4, 
                 warmup='exp', 
                 last_epoch=-1):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)
    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        """
        warmup 期间的学习率ratio
        """
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == "linear":
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == "exp":
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio

class WarmupPolyLrScheduler(WarmupLrScheduler):
    def __init__(self, 
                 optimizer, 
                 power, 
                 max_iter, 
                 warmup_iter=1000, 
                 warmup_ratio=5e-4, 
                 warmup='exp', 
                 last_epoch=-1):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio