from abc import ABC, abstractmethod
import math
import torch
from .log import log

# 基类 LRScheduler
class LRScheduler(ABC):
    @property
    @abstractmethod
    def cur_steps(self): ...  # 封装与只读：获取当前步数

    @property
    @abstractmethod
    def cur_lr(self): ...  # 封装与只读：获取当前学习率

    @abstractmethod
    def step(self): ...  # 抽象方法：执行一步更新

    @abstractmethod
    def can_clip_grad(self): ...  # 抽象方法：判断是否可以进行梯度裁剪

    @abstractmethod
    def get_ckpt_dict(self) -> dict: ...  # 抽象方法：获取用于保存检查点的状态字典

    @abstractmethod
    def restore_ckpt_dict(self, ckpt: dict): ...  # 抽象方法：从检查点字典恢复状态


# 继承自 LRScheduler 的具体实现类：带预热的余弦退火学习率调度器
class WarmupCosineAnnealingLRScheduler(LRScheduler):
    def __init__(
            self,
            *,
            optimizer: torch.optim.Optimizer,  # PyTorch 优化器实例
            warmup_iters: int,                 # 步数
            initial_lr: float,                 # 初始学习率（预热开始时的LR）
            min_lr: float,                     # 最小学习率（余弦退火的最低点）
            max_lr: float,                     # 最大学习率（预热结束/退火开始时的LR）
            cosine_annealing_period: int,      # 余弦退火的第一个周期的步数
            cosine_annealing_period_mul: int = 0, # 周期长度的倍增因子（SGDR算法），0表示无倍增
            need_log: bool = False             # 是否需要将学习率写入日志文件
    ):
        super().__init__()
        self.need_log = need_log
        
        self._optimizer = optimizer
        self._initial_lr = initial_lr
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._warmup_iters = warmup_iters
        self._cosine_annealing_period = cosine_annealing_period
        self._cosine_annealing_period_mul = cosine_annealing_period_mul

        self.T_cur = 0  # 初始化：当前余弦周期内已走过的步数
        self.cycle = 0  # 初始化：当前处于第几个余弦周期

        # 计算预热阶段每一步学习率的增量：线性增长
        if warmup_iters != 0:
            self._lr_increment = (max_lr - initial_lr) / warmup_iters  
        else:
            self._lr_increment = 0  # 如果没有预热，增量为0

        self._steps = -1  # 初始化全局总步数，从-1开始，第一次step后变为0
        self._current_lr = initial_lr  # 初始化当前学习率记录
        self._cosine_annealing_base_lr = None  # 用于记录进入余弦阶段时的基准学习率

        


    @property
    def cur_steps(self):
        return self._steps  # 返回当前的总步数

    @property
    def cur_lr(self):
        return self._current_lr  # 返回当前的学习率

    def step(self):
        self._steps += 1  # 总步数加1
        self._update_lr()  # 更新学习率

    def can_clip_grad(self):
        # 只有在预热阶段结束后，才允许进行梯度裁剪（通常策略）
        return self._steps > self._warmup_iters

    # 把计算出来的新学习率 lr，赋值给优化器管理的所有参数组，以便在下一次更新模型参数时生效
    def _update_lr(self):
        # 逻辑分支1：如果倍增因子为0（表示单周期模式），且总步数超过了（预热+周期长度）——>此时保持最小学习率
        if self._cosine_annealing_period_mul == 0 and self._steps >= self._cosine_annealing_period + self._warmup_iters:
            lr = self._min_lr  # 设定为最小学习率
            for param_group in self._optimizer.param_groups:  # 遍历优化器的参数组
                param_group['lr'] = lr  # 更新参数组的学习率
        
        # 逻辑分支2：如果处于预热阶段（当前步数 <= 预热步数）
        elif self._steps <= self._warmup_iters:
            # 线性调整学习率：初始值 + 当前步数 * 增量
            lr = self._initial_lr + self._steps * self._lr_increment
            for param_group in self._optimizer.param_groups:  # 遍历参数组
                param_group['lr'] = lr  # 更新学习率
        
        # 逻辑分支3：预热结束，进入余弦退火阶段
        else:
            # 基准学习率（通常是 max_lr）为空：第一次进入退火阶段
            if not self._cosine_annealing_base_lr:
                self._cosine_annealing_base_lr = self.cur_lr

            """每步更新学习率"""
            
            # 周期倍增因子mul
            # _cosine_annealing_period余弦退火的第一个周期的步数
            # 如果 mul > 0，SGDR 算法，周期长度会随 cycle 指数增长（例如：100, 200, 400...）
            # 如果 mul = 0 或 1，周期长度保持不变
            mul = max(self._cosine_annealing_period_mul, 1)
            # 计算当前周期的最大步数 T_max
            T_max = self._cosine_annealing_period * (mul ** self.cycle)

            # 更新当前周期内的步数计数器
            self.T_cur += 1
            # 检查是否完成了当前周期
            if self.T_cur >= T_max:
                self.cycle += 1  # 进入下一个周期
                self.T_cur = 0  # 重置周期内步数

            # 1. 计算进度比例 (0 到 1)
            #   如果上一个余弦退火周期已经完成了，这里就变0，重新从max_lr下降
            progress = self.T_cur / T_max 

            # 2. 放入余弦公式
                # math.pi * progress: 范围从 0 变到 π (3.14...)
                # math.cos(...): 范围从 1 (cos0) 变到 -1 (cosπ)
                # (1 + ...)/2: 把范围从 [1, -1] 压缩映射到 [1, 0]
            cos_factor = (1 + math.cos(math.pi * progress)) / 2
            
            # 根据余弦公式计算当前 LR
            # LR = min + (max - min) * factor
            lr = self._min_lr + (self._cosine_annealing_base_lr - self._min_lr) * cos_factor

            for param_group in self._optimizer.param_groups:  # 遍历参数组
                param_group['lr'] = lr  # 应用新的学习率

        self._current_lr = lr  # 更新类内部记录的当前学习率

        # 如果开启了日志记录，写入文件
        if self.need_log:
            log(f"step: {self.cur_steps}, lr: {lr}\n", 'lr.txt')

    def get_ckpt_dict(self) -> dict:
        # 返回当前状态的字典，用于保存检查点
        return {
            'cur_lr': self._current_lr,  # 当前学习率
            'lr_steps': self.cur_steps,  # 当前总步数
            'cosine_annealing_base_lr': self._cosine_annealing_base_lr, # 余弦基准学习率
            't_cur': self.T_cur,         # 当前周期内步数
            'cycle': self.cycle,         # 当前周期索引
        }

    def restore_ckpt_dict(self, ckpt: dict):
        # 从检查点字典恢复状态
        if 'cur_lr' in ckpt:
            self._current_lr = ckpt['cur_lr']  # 恢复当前学习率

        if 'lr_steps' in ckpt:
            self._steps = ckpt['lr_steps']  # 恢复总步数

        if 'cosine_annealing_base_lr' in ckpt:
            self._cosine_annealing_base_lr = ckpt['cosine_annealing_base_lr'] # 恢复基准学习率

        if 't_cur' in ckpt:
            self.T_cur = ckpt['t_cur']  # 恢复周期内步数

        if 'cycle' in ckpt:
            self.cycle = ckpt['cycle']  # 恢复周期索引

        self._update_lr()  # 恢复数据后，立即强制更新一次优化器中的 LR，确保同步


# 一个空的学习率调度器，不做任何调整
# 在trainer.py的_init_lr_scheduler方法中，当enable_lr_scheduler为False时返回NoneLRScheduler实例
class NoneLRScheduler(LRScheduler):
    def __init__(self, initial_lr):
        self._current_lr = initial_lr  # 只记录初始学习率，之后不再改变

    @property
    def cur_steps(self):
        return -1  # 不追踪步数，返回 -1

    @property
    def cur_lr(self):
        return self._current_lr  # 返回固定的学习率

    def step(self): ...  # 空方法，步进时不执行任何操作

    def can_clip_grad(self):
        return True  # 始终允许梯度裁剪

    def get_ckpt_dict(self) -> dict:
        return {'cur_lr': self._current_lr}  # 仅保存当前学习率

    def restore_ckpt_dict(self, ckpt: dict):
        if 'cur_lr' in ckpt:
            self._current_lr = ckpt['cur_lr']  # 仅恢复当前学习率