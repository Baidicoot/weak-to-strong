import json
import os
from typing import Dict, List, Optional, Sequence, Union

import fire
import numpy as np
import torch

import datasets

import weak_to_strong.logger as logger
from weak_to_strong.common import get_tokenizer
from weak_to_strong.datasets import (VALID_DATASETS, load_dataset,
                                     tokenize_dataset)
from weak_to_strong.loss import logconf_loss_fn, product_loss_fn, xent_loss
from weak_to_strong.train import ModelConfig, train_and_save_model, load_model
from weak_to_strong.eval import eval_model_acc

from copy import deepcopy

pythia_ckpts = list(range(1000, 144000, 1000))

MODEL_CONFIGS = [
    ModelConfig(
        name="EleutherAI/pythia-160m-v0",
        default_lr=5e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-410m-v0",
        default_lr=5e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-1b-v0",
        default_lr=5e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-1.4b-v0",
        default_lr=5e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-2.8b-v0",
        default_lr=5e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),
]

MODELS_DICT: Dict[str, ModelConfig] = {
    model_config.name: model_config for model_config in MODEL_CONFIGS
}

loss_dict = {
    "logconf": logconf_loss_fn(),
    "product": product_loss_fn(),
    "xent": xent_loss(),
}

VALID_LOSSES: List[str] = list(loss_dict.keys())

def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    transfer_loss: Union[str, Sequence[str]] = "xent,logconf",
    n_docs: int = 10000,
    n_test_docs: int = 200,
    weak_model_size: str = "EleutherAI/pythia-160m-v0",
    weak_lr: Optional[float] = None,
    strong_model_sizes: List[str] = [
        "EleutherAI/pythia-410m-v0",
        "EleutherAI/pythia-1b-v0",
        "EleutherAI/pythia-1.4b-v0",
    ],
    strong_lr: Optional[float] = None,
    # Defaults to strong_lr
    transfer_lr: Optional[float] = None,
    # Optims default to default_optimizer in the model definitions
    weak_optim: Optional[str] = None,
    strong_optim: Optional[str] = None,
    transfer_optim: Optional[str] = None,
    gt_epochs: int = 2,
    # defaults to gt_epochs
    transfer_epochs: Optional[int] = None,
    force_retrain: bool = False,
    seed: int = 0,
    minibatch_size_per_device: Optional[int] = None,
    train_with_dropout: bool = False,
    results_folder: str = "./ft_ckpts/results",
    linear_probe: bool = False,
    lr_schedule: str = "cosine_anneal",
    log_prefix: str = "",
    # Set to an absurdly high value so we don't do intermediate evals by default.
    eval_every: int = 100000000,
):
    # this is per device!
    if minibatch_size_per_device is None:
        minibatch_size_per_device = 32
    assert ds_name in VALID_DATASETS, f"Unknown dataset {ds_name} not in {VALID_DATASETS}"
    if isinstance(transfer_loss, str):
        transfer_losses = transfer_loss.split(",")
    else:
        transfer_losses = transfer_loss
    del transfer_loss
    for tloss in transfer_losses:
        assert tloss in VALID_LOSSES, f"Unknown loss {tloss} not in {VALID_LOSSES}"
    assert (
        weak_model_size in MODELS_DICT
    ), f"Unknown model size {weak_model_size} not in {MODELS_DICT}"
    weak_model_config = MODELS_DICT[weak_model_size]

    strong_model_configs = []
    for strong_model_size in strong_model_sizes:
        assert (
            strong_model_size in MODELS_DICT
        ), f"Unknown model size {strong_model_size} not in {MODELS_DICT}"
        strong_model_configs.append(MODELS_DICT[strong_model_size])

    if weak_lr is None:
        assert batch_size == 32
        weak_lr = weak_model_config.default_lr
    if transfer_epochs is None:
        transfer_epochs = gt_epochs

    if weak_optim is None:
        weak_optim = weak_model_config.default_optimizer
    if transfer_optim is None:
        transfer_optim = strong_optim

    weak_eval_batch_size = weak_model_config.eval_batch_size

    # Load dataset
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    split_data = train_dataset.train_test_split(test_size=0.5, seed=seed)
    train1_ds, train2_ds = split_data["train"], split_data["test"]
    print("len(train1):", len(train1_ds), "len(train2):", len(train2_ds))

    def train_model(
        model_config: ModelConfig,
        train_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        *,
        loss_type: str,
        label: str,
        subpath,
        lr,
        eval_batch_size,
        epochs=1,
        inference_ds: Optional[torch.utils.data.Dataset] = None,
        linear_probe: bool = False,
        optimizer_name: str = "adam",
    ):
        save_path = os.path.join(results_folder, subpath)
        linprobe_str = "_linprobe" if linear_probe else ""
        logger.configure(
            name="{log_prefix}{label}_{base_model_name}_{ds_name}_{loss_type}_{optimizer_name}_{lr}_{lr_schedule}{linprobe_str}_{datetime_now}",
            label=label,
            ds_name=ds_name,
            truncation_max_len=n_docs or "none",
            loss_type=loss_type,
            lr=lr,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            save_path=save_path,
            base_model_name=model_config.name,
            epochs=epochs,
            linprobe_str=linprobe_str,
            lr_schedule=lr_schedule,
            log_prefix=log_prefix,
            optimizer_name=optimizer_name,
        )
        # Tokenize datasets
        tokenizer = get_tokenizer(model_config.name)
        train_ds = tokenize_dataset(train_ds, tokenizer, max_ctx)
        test_ds = tokenize_dataset(test_ds, tokenizer, max_ctx)
        if inference_ds:
            inference_ds = tokenize_dataset(inference_ds, tokenizer, max_ctx)

        loss_fn = loss_dict[loss_type]
        return train_and_save_model(
            model_config,
            train_ds,
            test_ds,
            inference_ds=inference_ds,
            batch_size=batch_size,
            save_path=save_path,
            loss_fn=loss_fn,
            lr=lr,
            epochs=epochs,
            force_retrain=force_retrain,
            eval_batch_size=eval_batch_size,
            minibatch_size_per_device=minibatch_size_per_device,
            train_with_dropout=train_with_dropout,
            linear_probe=linear_probe,
            lr_schedule=lr_schedule,
            optimizer_name=optimizer_name,
            eval_every=eval_every,
        )

    print(f"Training weak model, size {weak_model_size}")
    weak_test_results, weak_ds = train_model(
        weak_model_config,
        train1_ds,
        test_ds,
        loss_type="xent",
        label="weak",
        subpath=os.path.join("weak_model_gt", weak_model_size.replace("/", "_")),
        lr=weak_lr,
        eval_batch_size=weak_eval_batch_size,
        inference_ds=train2_ds,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=weak_optim,
    )

    weak_test_acc = np.mean([r["acc"] for r in weak_test_results])

    closest_ckpts = {}

    for strong_model_config in strong_model_configs:
        print(f"Searching {strong_model_config.name} for checkpoint with accuracy closest to {weak_test_acc}")

        if strong_lr is None:
            assert batch_size == 32
            strong_lr = strong_model_config.default_lr
        if transfer_lr is None:
            transfer_lr = strong_lr
        if transfer_epochs is None:
            transfer_epochs = gt_epochs

        if strong_optim is None:
            strong_optim = strong_model_config.default_optimizer
        if transfer_optim is None:
            transfer_optim = strong_optim

        strong_eval_batch_size = strong_model_config.eval_batch_size

        tokenizer = get_tokenizer(strong_model_config.name)
        test_ds = tokenize_dataset(test_ds, tokenizer, 1024)

        interval = pythia_ckpts

        endpoint_acc = [None, None]

        cfg = deepcopy(strong_model_config)
        cfg.checkpoint = interval[0]

        start_test_results, _ = train_model(
            cfg,
            train1_ds,
            test_ds,
            loss_type="xent",
            label="strong",
            subpath=os.path.join("strong_model_gt", strong_model_size.replace("/", "_") + "_step" + str(cfg.checkpoint)),
            lr=strong_lr,
            eval_batch_size=strong_eval_batch_size,
            inference_ds=train2_ds,
            epochs=gt_epochs,
            linear_probe=linear_probe,
            optimizer_name=strong_optim,
        )

        endpoint_acc[0] = np.mean([r["acc"] for r in start_test_results])

        print(f"Start accuracy: {endpoint_acc[0]}")

        cfg = deepcopy(strong_model_config)
        cfg.checkpoint = interval[-1]

        end_test_results, _ = train_model(
            cfg,
            train1_ds,
            test_ds,
            loss_type="xent",
            label="strong",
            subpath=os.path.join("strong_model_gt", strong_model_size.replace("/", "_") + "_step" + str(cfg.checkpoint)),
            lr=strong_lr,
            eval_batch_size=strong_eval_batch_size,
            inference_ds=train2_ds,
            epochs=gt_epochs,
            linear_probe=linear_probe,
            optimizer_name=strong_optim,
        )

        endpoint_acc[1] = np.mean([r["acc"] for r in end_test_results])

        del end_test_results

        print(f"End accuracy: {endpoint_acc[1]}")

        while len(interval) > 2:
            mid = len(interval) // 2

            cfg = deepcopy(strong_model_config)
            cfg.checkpoint = interval[mid]

            mid_test_results, _ = train_model(
                cfg,
                train1_ds,
                test_ds,
                loss_type="xent",
                label="strong",
                subpath=os.path.join("strong_model_gt", strong_model_size.replace("/", "_") + "_step" + str(cfg.checkpoint)),
                lr=strong_lr,
                eval_batch_size=strong_eval_batch_size,
                inference_ds=train2_ds,
                epochs=gt_epochs,
                linear_probe=linear_probe,
                optimizer_name=strong_optim,
            )

            mid_acc = np.mean([r["acc"] for r in mid_test_results])

            print(mid_acc, weak_test_acc)

            if mid_acc > weak_test_acc:
                interval = interval[:mid]
                endpoint_acc[1] = mid_acc
            else:
                interval = interval[mid:]
                endpoint_acc[0] = mid_acc
        
        best = None

        if endpoint_acc[1] - weak_test_acc < weak_test_acc - endpoint_acc[0]:
            best = interval[1]
        else:
            best = interval[0]
        print(f"Found closest {best} for {strong_model_config.name}")
        closest_ckpts[strong_model_config.name] = best
    
    print(closest_ckpts)
    
if __name__ == "__main__":
    fire.Fire(main)