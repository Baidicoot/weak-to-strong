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

from weak_to_strong.model import TransformerWithHead

from copy import deepcopy

pythia_ckpts = list(range(1000, 144000, 1000))

MODEL_CONFIGS = [
    ModelConfig(
        name="EleutherAI/pythia-160m-v0",
        default_lr=1e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-410m-v0",
        default_lr=1e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-1b-v0",
        default_lr=1e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-1.4b-v0",
        default_lr=1e-5,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="EleutherAI/pythia-2.8b-v0",
        default_lr=1e-4,
        eval_batch_size=32,
        model_parallel=True,
    ),

    ModelConfig(
        name="gpt2",
        default_lr=1e-4,
        eval_batch_size=32,
        model_parallel=True,
    )
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

def main_2():
    model_cfg = MODELS_DICT["gpt2"]

    model = TransformerWithHead.from_pretrained(
        model_cfg.name,
        device_map = "auto",
    )

    model.save_pretrained("./ft_ckpts/test")

    print("saved")

    model = TransformerWithHead.from_pretrained(
        "./ft_ckpts/test",
        device_map = "auto",
    )

def main(
    batch_size: int = 32,
    max_ctx: int = 1024,
    ds_name: str = "sciq",
    transfer_loss: Union[str, Sequence[str]] = "xent,logconf",
    n_docs: int = 1000,
    n_test_docs: int = 1000,
    model_size: str = "EleutherAI/pythia-160m-v0",
    lr: Optional[float] = None,
    optim: Optional[str] = None,
    gt_epochs: int = 2,
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
    eval_batch_size = batch_size

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
    
    dataset = load_dataset(ds_name, seed=seed, split_sizes=dict(train=n_docs, test=n_test_docs))

    # Split the training dataset in half
    train_dataset, test_ds = dataset["train"], dataset["test"]

    model_config = MODELS_DICT[model_size]

    lr = lr or model_config.default_lr
    optim = optim or model_config.default_optimizer

    test_results, _ = train_model(
        model_config,
        train_dataset,
        test_ds,
        loss_type="xent",
        label="test_model",
        subpath="test_model",
        lr=lr,
        eval_batch_size=eval_batch_size,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=optim,
    )

    test_2_results, _ = train_model(
        model_config,
        train_dataset,
        test_ds,
        loss_type="xent",
        label="test_model",
        subpath="test_model",
        lr=lr,
        eval_batch_size=eval_batch_size,
        epochs=gt_epochs,
        linear_probe=linear_probe,
        optimizer_name=optim,
    )

    assert test_results == test_2_results

if __name__ == "__main__":
    fire.Fire(main)