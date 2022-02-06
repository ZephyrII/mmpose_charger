# Copyright (c) OpenMMLab. All rights reserved.
import warnings


from functools import partial
from mmcv.utils.parrots_wrapper import _get_dataloader
from mmpose.datasets.builder import worker_init_fn

from torch.utils.data.sampler import Sampler, RandomSampler

import numpy as np

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, get_dist_info

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataset, DistributedSampler, build_dataloader
from mmpose.utils import get_root_logger

from mmpose.apis.test import single_gpu_val

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook

# class RandomSampler(Sampler):
#     def __init__(self, data_source, num_samples=None):
#         self.data_source = data_source
#         self._num_samples = num_samples

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError(
#                 "num_samples should be a positive integer "
#                 "value, but got num_samples={}".format(self.num_samples)
#             )

#     @property
#     def num_samples(self):
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self):
#         n = len(self.data_source)
        
#         return iter(np.random.choice(n, self.num_samples))
#         # return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

#     def __len__(self):
#         return self.num_samples

def build_dataloader(
    dataset,
    samples_per_gpu: int,
    workers_per_gpu: int,
    num_gpus: int = 1,
    dist: bool = True,
    shuffle: bool = True,
    seed: int = None,
    drop_last: bool = True,
    pin_memory: bool = True,
    limited_val: bool = False,
    **kwargs,
):
    """Based on https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/builder.py#L46
    Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: True
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed
        )
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # if limited_val:
        #     shuffle = False
        #     sampler = RandomSampler(dataset, num_samples=40, replacement=True)
        # else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
    # Force shorter epoch
    # shuffle = False
    # sampler = RandomSampler(dataset, num_samples=10, replacement=True)

    init_fn = (
        partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    _, DataLoader = _get_dataloader()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs,
    )

    return data_loader



def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # determine wether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, test_fn=single_gpu_val, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
