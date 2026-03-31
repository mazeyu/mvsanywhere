""" 
    Trains a DepthModel model. Uses an MVS dataset from datasets.

    - Outputs logs and checkpoints to opts.log_dir/opts.name
    - Supports mixed precision training by setting '--precision 16'

    We train with a batch_size of 16 with 16-bit precision on two A100s.

    Example command to train with two GPUs
        python train.py --name HERO_MODEL \
                    --log_dir logs \
                    --config_file configs/models/hero_model.yaml \
                    --data_config configs/data/scannet_default_train.yaml \
                    --gpus 2 \
                    --batch_size 16;
                    
"""


import os
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, Strategy
from torch.utils.data import DataLoader, ConcatDataset

import mvsanywhere.options as options
from mvsanywhere.utils.dataset_utils import get_dataset
from mvsanywhere.utils.generic_utils import copy_code_state
from mvsanywhere.utils.model_utils import get_model_class
from mvsanywhere.experiment_modules.rmvd_mvsa import MVSA_Wrapped
from tqdm import tqdm
import skimage
import numpy as np
import torch.nn as nn


def _inputs_and_gt_from_sample(sample, inputs=["images", "intrinsics", "poses", "depth_range"]):
    is_input = lambda key: key in inputs or key == "keyview_idx"
    sample_inputs = {key: val for key, val in sample.items() if is_input(key)}
    sample_gt = {key: val for key, val in sample.items() if not is_input(key)}
    return sample_inputs, sample_gt


def _compute_metrics(sample_gt, pred):
    from rmvd.eval.metrics import m_rel_ae, pointwise_rel_ae, thresh_inliers, sparsification
    gt_depth = sample_gt['depth'][0, 0]
    pred_depth = pred['depth'][0, 0]
    eval_mask = np.ones_like(pred_depth, dtype=bool)
    absrel = m_rel_ae(gt=gt_depth, pred=pred_depth, mask=eval_mask, output_scaling_factor=100.0)
    metrics = {'absrel': absrel}
    for i in [30]:
        thresh = 1.03 ** (i / 30)
        inliers = thresh_inliers(gt=gt_depth, pred=pred_depth, thresh=thresh, mask=eval_mask, output_scaling_factor=100.0)
        metrics[f"inliers_{i}"] = inliers
    return metrics


def _postprocess_sample_and_output(sample_inputs, sample_gt, pred, clip_pred_depth=True):
    gt_depth = sample_gt['depth']
    pred_depth = pred['depth']
    pred_depth = skimage.transform.resize(pred_depth, gt_depth.shape, order=0, anti_aliasing=False)

    pred_mask = np.ones_like(pred_depth, dtype=bool)
    gt_mask = gt_depth > 0
    if isinstance(clip_pred_depth, tuple):
        pred_depth = np.clip(pred_depth, clip_pred_depth[0], clip_pred_depth[1]) * pred_mask
    elif clip_pred_depth:
        pred_depth = np.clip(pred_depth, 0.1, 100) * pred_mask

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_invdepth = np.nan_to_num(1 / pred_depth, nan=0, posinf=0, neginf=0)

    if 'depth_uncertainty' in pred:
        pred_depth_uncertainty = pred['depth_uncertainty']
        pred_depth_uncertainty = skimage.transform.resize(pred_depth_uncertainty, gt_depth.shape, order=0,
                                                            anti_aliasing=False)
        pred['depth_uncertainty'] = pred_depth_uncertainty

    pred['depth'] = pred_depth
    pred['invdepth'] = pred_invdepth


def deep_clone(obj):
    if torch.is_tensor(obj):
        return obj.clone().detach()
    elif isinstance(obj, np.ndarray):
        return obj.copy()
    elif isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_clone(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_clone(v) for v in obj)
    else:
        return obj


class RMVDEvaluationCallback(pl.Callback):
    def __init__(self, opts):
        self.opts = opts
        self._last_trigger_step = -1

    def _run_rmvd_eval(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        model = MVSA_Wrapped(self.opts, use_refinement=True, model=pl_module)
        rmvd_val_dir = os.environ.get("RMVD_SAMPLES_VAL_DIR")
        if rmvd_val_dir is None:
            return

        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            for dataset in ["eth3d", "kitti", "dtu", "scannet", "tanks_and_temples"]:
                sample_root = os.path.join(rmvd_val_dir, dataset)
                if not os.path.exists(sample_root):
                    continue
                absrels = []
                inliers = []
                for sample_id in tqdm(sorted(os.listdir(sample_root))):
                    sample_path = os.path.join(sample_root, sample_id)
                    sample = torch.load(sample_path)
                    sample_inputs0, sample_gt = _inputs_and_gt_from_sample(sample)
                    if len(sample_gt["depth"].shape) != 4:
                        sample_gt["depth"] = np.expand_dims(sample_gt["depth"], axis=0)

                    metrics = {}
                    n = len(sample_inputs0["images"]) - 1

                    if self.opts.model_num_views == 8:
                        pair_candidates = [
                            [0, *range(1, n // 2 + 1)],
                            [0, *range(n // 2 + 1, n + 1)],
                            [0, *range(1, n + 1)],
                        ]
                    else:
                        pair_candidates = [[0, i] for i in range(1, n + 1, 2)]

                    for i, frames in enumerate(pair_candidates):
                        original_frames = [sample_inputs0['keyview_idx'][0], *[j for j in range(n + 1) if j != sample_inputs0['keyview_idx'][0]]]
                        sample_inputs = deep_clone(sample_inputs0)
                        sample_inputs["images"] = [sample_inputs["images"][original_frames[j]] for j in frames]
                        sample_inputs["poses"] = [sample_inputs["poses"][original_frames[j]] for j in frames]
                        sample_inputs["intrinsics"] = [sample_inputs["intrinsics"][original_frames[j]] for j in frames]
                        sample_inputs['keyview_idx'][0] = 0
                        sample_inputs = model.input_adapter(**sample_inputs)
                        pred = model.output_adapter(model(**sample_inputs))[0]
                        _postprocess_sample_and_output(sample_inputs, sample_gt, pred)
                        metrics[i] = _compute_metrics(sample_gt, pred)
                    best_i = max(metrics.keys(), key=lambda x: metrics[x]['inliers_30'])
                    absrels.append(metrics[best_i]['absrel'])
                    inliers.append(metrics[best_i]['inliers_30'])
                absrel = np.mean(absrels)
                inlier = np.mean(inliers)
                for key, val in zip(["absrel", "inliers_103"], [absrel, inlier]):
                    trainer.logger.experiment.add_scalar(f"custom_eval/{dataset}_{key}", val, global_step=trainer.global_step)

        if was_training:
            pl_module.train()

    def on_validation_end(self, trainer, pl_module):
        self._run_rmvd_eval(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        num_val_batches = trainer.num_val_batches
        if isinstance(num_val_batches, (list, tuple)):
            has_real_val_data = any(v > 0 for v in num_val_batches)
        else:
            has_real_val_data = num_val_batches > 0

        if has_real_val_data:
            return

        if not isinstance(self.opts.val_interval, int) or self.opts.val_interval <= 0:
            return

        step = trainer.global_step
        if step <= 0 or step % self.opts.val_interval != 0:
            return

        if step == self._last_trigger_step:
            return

        self._last_trigger_step = step
        self._run_rmvd_eval(trainer, pl_module)


def prepare_dataloaders(opts: options.Options) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Prepare training and validation dataloaders.
    Training loader is one, while we might have multiple dataloaders for validations.
    For instance, we might validate using a different augmentation for hints (always given, never
    given, given with 50% chances etc).

    Params:
        opts: options for the current run
    Returns:
        a train dataloader, a validation dataloader (or None if no validation set)
    """
    train_datasets, val_datasets = [], []
    for dataset in opts.datasets:
        dataset_class, _ = get_dataset(
            dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
        )

        train_dataset = dataset_class(
            dataset.dataset_path,
            split="train",
            mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
            num_images_in_tuple=opts.num_images_in_tuple,
            tuple_info_file_location=dataset.tuple_info_file_location,
            image_width=opts.image_width,
            image_height=opts.image_height,
            shuffle_tuple=opts.shuffle_tuple,
            matching_scale=opts.matching_scale,
            prediction_scale=opts.prediction_scale,
            prediction_num_scales=opts.prediction_num_scales
        )
        train_datasets.append(train_dataset)


    if opts.val_datasets:
        for dataset in opts.val_datasets:
            dataset_class, _ = get_dataset(
                dataset.dataset, dataset.dataset_scan_split_file, opts.single_debug_scan_id
            )
            val_dataset = dataset_class(
                dataset.dataset_path,
                split="val",
                mv_tuple_file_suffix=dataset.mv_tuple_file_suffix,
                num_images_in_tuple=opts.num_images_in_tuple,
                tuple_info_file_location=dataset.tuple_info_file_location,
                image_width=opts.val_image_width,
                image_height=opts.val_image_height,
                include_full_res_depth=opts.high_res_validation,
                matching_scale=opts.matching_scale,
                prediction_scale=opts.prediction_scale,
                prediction_num_scales=opts.prediction_num_scales
            )
            val_datasets.append(val_dataset)

    train_dataloader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    # Only create validation dataloader if there are validation datasets
    if val_datasets:
        val_dataloader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=opts.val_batch_size,
            shuffle=False,
            num_workers=opts.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def prepare_callbacks(
    opts: options.Options, enable_version_counter: bool = True, is_resume: bool = False
) -> List[pl.pytorch.callbacks.Callback]:
    """Prepare callbacks for the training.
    In our case, callbacks are the strategy used to save checkpoints during training and the
    learning rate monitoring.

    Params:
        opts: options for the current run
        enable_version_counter: if True, save checkpoints with lightning versioning
    Returns:
        a list of callbacks
    """
    # set a checkpoint callback for lignting to save model checkpoints
    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        verbose=True,
        monitor="val_metrics/a5",
        mode="max",
        dirpath=str((Path(opts.log_dir) / opts.name).resolve()),
    )

    # keep track of changes in learning rate
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor, RMVDEvaluationCallback(opts)]
    return callbacks


def prepare_model(opts: options.Options) -> torch.nn.Module:
    """Prepare model to train.
    The function selects the right model given the model class, and eventually resumes the model
    from a checkpoint if `load_weights_from_checkpoint` or `lazy_load_weights_from_checkpoint`
    are set.

    Params:
        opts: options for the current run
    Returns:
        (resumed) model to train
    """
    model_class_to_use = get_model_class(opts)

    if opts.load_weights_from_checkpoint is not None:
        model = model_class_to_use.load_from_checkpoint(
            opts.load_weights_from_checkpoint,
            opts=opts,
            args=None,
        )
    elif opts.lazy_load_weights_from_checkpoint is not None:
        model = model_class_to_use(opts)
        state_dict = torch.load(opts.lazy_load_weights_from_checkpoint)["state_dict"]
        available_keys = list(state_dict.keys())
        for param_key, param in model.named_parameters():
            if param_key in available_keys:
                try:
                    if isinstance(state_dict[param_key], torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = state_dict[param_key].data
                    else:
                        param = state_dict[param_key]

                    model.state_dict()[param_key].copy_(param)
                    print('Param copied: ', param_key)
                except:
                    print(f"WARNING: could not load weights for {param_key}")
    else:
        # load model using read options
        model = model_class_to_use(opts)
    return model


def prepare_ddp_strategy(opts: options.Options) -> Strategy:
    """Prepare the strategy for data parallel. It defines how to manage multiple processes
    over one or multiple nodes.

    Params:
        opts: options for the current run
    Returns:
        data parallel strategy
    """
    # allowing the lightning DDPPlugin to ignore unused params.
    find_unused_parameters = (opts.matching_encoder_type == "unet_encoder") or ("dinov2" in opts.image_encoder_name) or ("depth_anything" in opts.depth_decoder_name)
    return DDPStrategy(find_unused_parameters=find_unused_parameters)


def prepare_trainer(
    opts: options.Options,
    logger: pl.pytorch.loggers.logger.Logger,
    callbacks: List[pl.pytorch.callbacks.Callback],
    ddp_strategy: Strategy,
    plugins: List[pl.pytorch.plugins._PLUGIN_INPUT] = None,
    resume_ckpt: Optional[str] = None,
    auto_devices: bool = False,
) -> pl.pytorch.trainer.trainer.Trainer:
    """
    Prepare a trainer for the run.
    Params:
        opts: options for the current run
        logger: selected pl logger to use for logging
        callbacks: callbacks for the trainer (such as LRMonitor, Checkpoint saving strategy etc)
        ddp_strategy: strategy for data parallel plugins
        plugins: optional plugins in case of clusters. Default is none because we use a single machine
    Returns:
        (resumed) model to train
    """
    devices = "auto" if auto_devices else opts.gpus

    trainer = pl.Trainer(
        devices=devices,
        log_every_n_steps=opts.log_interval,
        val_check_interval=opts.val_interval,
        limit_val_batches=opts.val_batches,
        max_steps=opts.max_steps,
        precision=opts.precision,
        benchmark=True,
        logger=logger,
        sync_batchnorm=False,
        callbacks=callbacks,
        num_sanity_val_steps=opts.num_sanity_val_steps,
        strategy=ddp_strategy,
        plugins=plugins,
        limit_train_batches=10000,
        profiler="simple",
        check_val_every_n_epoch=None,
    )
    return trainer


def main(opts):
    # set seed
    pl.seed_everything(opts.random_seed)

    # prepare model
    model = prepare_model(opts=opts)

    # prepare dataloaders
    train_dataloader, val_dataloaders = prepare_dataloaders(opts=opts)

    # set up a tensorboard logger through lightning
    logger = TensorBoardLogger(save_dir=opts.log_dir, name=opts.name)

    # This will copy a snapshot of the code (minus whatever is in .gitignore)
    # into a folder inside the main log directory.
    copy_code_state(path=os.path.join(logger.log_dir, "code"))

    # dumping a copy of the config to the directory for easy(ier)
    # reproducibility.
    options.OptionsHandler.save_options_as_yaml(
        os.path.join(logger.log_dir, "config.yaml"),
        opts,
    )

    # prepare ddp strategy
    ddp_strategy = prepare_ddp_strategy(opts=opts)

    # prepare callbacks
    callbacks = prepare_callbacks(opts=opts)

    # prepare trainer
    trainer = prepare_trainer(
        opts=opts,
        logger=logger,
        callbacks=callbacks,
        ddp_strategy=ddp_strategy,
    )

    # start training
    trainer.fit(model, train_dataloader, val_dataloaders, ckpt_path=opts.resume)


if __name__ == "__main__":
    # get an instance of options and load it with config file(s) and cli args.
    option_handler = options.OptionsHandler()
    option_handler.parse_and_merge_options()
    option_handler.pretty_print_options()
    print("\n")
    opts = option_handler.options

    # if no GPUs are available for us then, use the 32 bit on CPU
    if opts.gpus == 0:
        print("Setting precision to 32 bits since --gpus is set to 0.")
        opts.precision = 32

    opts.lr_steps = [int(0.7 * opts.max_steps * 0.6665 / 2), int(0.8 * opts.max_steps * 0.6665 / 2)]

    main(opts)
