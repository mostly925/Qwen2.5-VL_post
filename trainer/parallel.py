import os
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # 导入分布式采样器，用于多卡数据切分
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入 DDP 模块

try:
    import deepspeed
except: ...  # 如果导入失败（未安装 DeepSpeed），则忽略错误，不做处理

from .log import log


class Parallel(ABC):  # 父类
    def __init__(
            self,
            init_process_group: bool = True,  # 是否初始化分布式进程组，默认为 True
            use_parallel: bool = True,  # 是否使用并行训练，默认为 True
            use_compile: bool = False  # 是否使用 torch.compile：编译优化有利于提升效率，比如算子融合、硬件适配等
    ):
        self._initialize(init_process_group, use_parallel, use_compile)

    def _initialize(
            self,
            init_process_group: bool,
            use_parallel: bool,
            use_compile: bool
    ):
        # 从环境变量获取全局 rank（进程 ID），如果不存在则默认为 -1
        self._global_rank: int = int(os.environ.get('RANK', -1))
        # 从环境变量获取当前节点上的 GPU ID，如果不存在则默认为 -1
        self._local_rank: int = int(os.environ.get('LOCAL_RANK', -1))
        # 判断是否真正启用并行：用户开启并行且环境变量中存在有效的 rank
        self._use_parallel: bool = use_parallel and self._global_rank != -1
        # 是否使用编译模式
        self._use_compile = use_compile

        # 初始化采样器为 None
        self._sampler: Optional[DistributedSampler] = None

        # 初始化模型占位符，model 通常指包装后的模型（如 DDP）
        self.model: Optional[nn.Module] = None
        # 初始化原始模型占位符，指向未被包装的 model
        self.raw_model: Optional[nn.Module] = None

        # 如果启用编译模式，设置 float32 的矩阵乘法精度为 'high' (使用 TF32，加速训练)
        if use_compile:
            torch.set_float32_matmul_precision('high')

        # 如果启用并行训练
        if self._use_parallel:
            # 如果需要初始化进程组
            if init_process_group:
                # 初始化分布式进程组，使用 NCCL 后端（NVIDIA GPU 通信标准）
                dist.init_process_group(backend='nccl')

            # 根据本地 rank 构造设备字符串，例如 'cuda:0'
            self.device: str = f'cuda:{self._local_rank}'
            # 设置设备类型为 cuda
            self.device_type: str = 'cuda'

            # 强制设置当前进程使用的 CUDA 设备，防止张量分配到错误的 GPU 上
            torch.cuda.set_device(self.device)

            #                          全局 rank             当前 rank                        总进程数
            log(f'global_rank={self._global_rank}, local_rank={self._local_rank}, world_size={self.world_size}')
        else:  # 如果不使用并行训练（单机单卡或 CPU）
            device = "cpu"  # 默认设备为 CPU
            if torch.cuda.is_available():  # 如果 CUDA 可用
                device = "cuda"  # 设备设为 cuda

            # 保存设备名称
            self.device: str = device
            # 保存设备类型
            self.device_type: str = device


    @abstractmethod  # 定义抽象方法，功能靠子类实现
    def process(
            self,
            model: nn.Module,  # 待处理的模型
            optimizer: torch.optim.Optimizer,  # 优化器
            kwargs: Optional[dict] = None,  # 额外的配置参数
            save_instance: bool = True  # 是否在实例中保存模型引用
    ) -> Tuple[nn.Module, torch.optim.Optimizer]: ...  # 返回处理后的模型和优化器

    def process_dataloader(
            self,
            dataset: Dataset,  # 数据集
            data_loader_kwargs: dict,  # 传给 DataLoader 的参数（如 batch_size）
            sampler_kwargs: Optional[dict]=None  # 传给 DistributedSampler 的参数
    ) -> DataLoader:
        """
        处理 DataLoader，如果是并行模式则自动添加分布式采样器。
        data_loader_kwargs: DataLoader 的关键字参数
                "batch_size" int,
                "pin_memory" bool,
                "collate_fn" collate_fn,
                "num_workers" int
                "shuffle" bool
                "drop_last" bool
        sampler_kwargs: 采样器 的关键字参数
                "shuffle" bool
                "drop_last" bool
        return: 构造好的 DataLoader
        """

       # 如果使用并行训练
        if self._use_parallel:
            # 实例化分布式采样器，确保不同 GPU 拿到不同的数据切片
            # 注意：sampler_kwargs 里面包含了 shuffle=True，这会让 Sampler 负责打乱
            self._sampler = DistributedSampler(dataset=dataset, **sampler_kwargs)
            
            # [关键修复] 创建一个新的参数字典，并移除 shuffle
            # 因为 DataLoader 不允许同时存在 sampler 和 shuffle=True
            dl_kwargs = data_loader_kwargs.copy()
            if 'shuffle' in dl_kwargs:
                del dl_kwargs['shuffle']

            # 返回带有采样器的 DataLoader，使用修改后的 dl_kwargs
            return DataLoader(dataset=dataset, sampler=self._sampler, **dl_kwargs)

        # 如果不是并行训练，直接返回普通的 DataLoader (此时可以使用 shuffle=True)
        return DataLoader(dataset=dataset, **data_loader_kwargs)

    def on_epoch_start(self, epoch):
        # 在每个 epoch 开始时调用
        if self._sampler:
            # 告诉采样器这是第几轮，以便它更换随机种子重新洗牌
            # DistributedSampler 的作用是将数据集切分成 N 份，DistributedSampler 在打乱数据索引时，依赖一个种子
            # 随机种子=seed + epoch    如果不传入epoch，每个批次都是同样的
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch): ...  # epoch 结束时的钩子函数，默认为空

    def synchronize(self):
        # 同步函数
        if self._use_parallel:
            # 等待当前设备上的所有 CUDA 完成任务，确保同步
            torch.cuda.synchronize(device=self.device)

    def destroy(self):
        # 销毁资源
        if self._use_parallel:
            # 销毁分布式进程组，释放资源
            dist.destroy_process_group()

    @property
    def parallel_train(self) -> bool:
        # 属性：是否并行训练
        return self._use_parallel

    @property
    def is_main_process(self) -> bool:
        # 属性：判断当前是否为主进程（rank 0）
        if self._use_parallel:
            return self._global_rank == 0  # 只有 rank 0 返回 True

        return True  # 非并行模式，唯一进程即为主进程

    @property
    def world_size(self) -> int:
        # 属性：获取总进程数（显卡总数）
        if self._use_parallel:
            return dist.get_world_size()
        return 1  # 非并行模式，world_size 为 1

    def wait(self, msg=None):
        # 等待所有进程到达此同步点
        if self.world_size == 1:  # 如果是单卡，不需要等待
            return
        # 打印等待日志
        log(f'wait at {self.device} for {msg}')
        dist.barrier()# 阻塞直到所有进程都运行到这行代码
        # 打印继续执行日志
        log(f'continue at {self.device}{msg}')


class DsParallel(Parallel):  # DeepSpeed 并行实现类
    def __init__(self):
        # 使用 NCCL（NVIDIA GPU 通信标准）初始化 DeepSpeed 分布式环境：
        # 获取 self._global_rank (当前是第几个进程)
        # 获取 self._local_rank (当前在第几张显卡)
        # 设置 self.device (如 'cuda:0')
        # 执行 torch.cuda.set_device (绑定当前显卡)
        deepspeed.init_distributed(dist_backend='nccl')
        # 调用父类初始化
        # PyTorch 分布式训练的硬性规则：在一个进程中，分布式通信组（Process Group）只能被初始化一次
        # 上面init_distributed()时候已经调用了 PyTorch 的 dist.init_process_group() 来建立 GPU 之间的通信连接
        # 在此就不需要基类Parallel再做了
        super().__init__(init_process_group=False)

    # 子类实现process方法
    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,  # DeepSpeed 的配置参数 (config_params)
            save_instance: bool = True   # 是否保存模型实例
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        return: 处理后的模型和优化器
        """

        # 如果需要保存实例，先保存原始模型
        if save_instance:
            self.raw_model = model

        # 使用 DeepSpeed 初始化模型引擎
        # deepspeed.initialize 返回 engine, optimizer, dataloader, scheduler
        model, optim, _, _ = deepspeed.initialize(
            model=model,  # 传入模型
            optimizer=optimizer,  # 传入优化器
            dist_init_required=False,  # 已经手动初始化过 distributed，设为 False
            config_params=kwargs  # 传入 DeepSpeed 配置文件或字典
        )

        # 如果需要保存实例，保存 DeepSpeed 包装后的 engine 为 self.model
        if save_instance:
            self.model = model

        # 返回 DeepSpeed 引擎和优化器
        return model, optim

    def synchronize(self): ...  # DeepSpeed 通常不需要手动调用 cuda.synchronize

    def destroy(self): ...  # DeepSpeed 销毁逻辑通常由库内部或脚本结束时处理


class DdpParallel(Parallel):  # PyTorch 原生 DDP 并行实现类
    def __init__(self):
        # 调用父类初始化（会执行 dist.init_process_group分布式通信分组）
        super().__init__()
    # 子类实现process方法
    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        
        model.to(self.device)
        # 如果启用编译模式
        if self._use_compile:
            # 使用 torch.compile 优化模型
            model = torch.compile(model)

        # 如果使用并行训练
        if self._use_parallel:
            # 初始化 DDP 包装器
            #                         device_ids 指定当前卡             output_device 指定输出设备，防止多卡显存溢出到卡0
            model = DDP(module=model, device_ids=[self._local_rank], output_device=self._local_rank)
            # 若不手动指定output_device，DDP默认行为是：将所有进程的输出数据（如模型前向输出、参数同步数据等）集中到卡 0
            # 获取 DDP 包装下的原始模型（module 属性）
            raw_model = model.module
        else:
            # 非并行模式，保持原样
            model = model
            raw_model = model

        # 如果需要保存实例
        if save_instance:
            self.model = model  # 保存（DDP 包装的）模型
            self.raw_model = raw_model  # 保存原始模型

        # 返回模型和优化器
        return model, optimizer


class NoneParallel(Parallel):  # 单机实现类
    def __init__(self):
        # 调用父类初始化，强制关闭 use_parallel
        super().__init__(use_parallel=False)
    # 子类实现process方法
    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        # 将模型移动到计算设备（CPU/GPU）
        model.to(self.device)

        # 如果启用编译模式
        if self._use_compile:
            # 使用 torch.compile 优化模型
            model = torch.compile(model)

        # 如果需要保存实例
        if save_instance:
            self.raw_model = model  # 当前模型
            self.model = model  # 当前模型

        # 返回模型和优化器
        return model, optimizer