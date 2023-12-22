# adapted from https://github.com/EleutherAI/elk-generalization/blob/main/elk_generalization/training/sft.py

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import os
import shutil

import numpy as np

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig  # type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

from typing import Callable, TypeVar, Optional, Type, Any, cast, Tuple, List

from weak_to_strong.eval import eval_model_acc

from sklearn.metrics import roc_auc_score

class LastTokenOnlyDataCollator(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: list[list[int] | Any | dict[str, Any]]
    ) -> dict[str, Any]:
        batch = super().torch_call(examples)
        # Compute the sequence length of each sample in the batch
        seq_lens = torch.sum(batch["input_ids"] != 0, dim=1)

        # Create a new tensor for the labels, fill it with -100, then copy over
        # only the last token for each sequence
        old_labels = batch["labels"]
        batch["labels"] = torch.full_like(old_labels, -100).scatter_(
            1, seq_lens[:, None] - 1, old_labels.gather(1, seq_lens[:, None] - 1)
        )

        return batch

def balance(ds: Dataset) -> Dataset:
    """Balance a dataset by undersampling the majority class."""
    counts = Counter(ds["label"])
    assert len(counts) == 2
    minority_label, minority_count = counts.most_common()[1]
    majority_label, _ = counts.most_common()[0]
    minority_ds = ds.filter(lambda x: x["label"] == minority_label)
    majority_ds = ds.filter(lambda x: x["label"] == majority_label).shuffle(42)

    return concatenate_datasets(
        [minority_ds, majority_ds.select(range(minority_count))]
    ).shuffle(42)

T = TypeVar("T")

def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)

In, Out = TypeVar("In"), TypeVar("Out")
DictFn = Callable[[dict[str, In]], dict[str, Out]]
VmappedFn = Callable[[dict[str, list[In]]], dict[str, list[Out]]]

def dict_vmap(func: DictFn) -> VmappedFn:
    """Turn a function taking dict[str, In] into one that takes dict[str, list[In]]."""

    def wrapper(input_dict: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # Transpose the input dict of lists into a list of dicts
        keys = input_dict.keys()
        transposed_input = [
            dict(zip(keys, values)) for values in zip(*input_dict.values())
        ]

        # Apply the wrapped function to each dict
        results = [func(single_input) for single_input in transposed_input]

        # Transpose the results back into a dict of lists
        # Assuming that each result is a dictionary
        transposed_output = {
            key: [result[key] for result in results] for key in results[0]
        }

        return transposed_output

    return wrapper

def get_model_path(
    model_name: str,
    label: str,
    save_path: Optional[str],
    model_ckpt: Optional[int] = None,
) -> Optional[str]:
    if save_path is None:
        return None

    model_short = model_name.split("/")[-1]

    if model_ckpt is None:
        return f"{save_path}/{label}/{model_short}"
    else:
        return f"{save_path}/{label}/{model_short}_step{model_ckpt}"

def get_neox_lora_modules(
    model_name: str,
    mlps: bool = True,
    attns: bool = True,
):
    config = AutoConfig.from_pretrained(model_name)

    mlps = [
        f"gpt_neox.layers.{layer}.mlp.dense_h_to_4h"
        for layer in range(config.num_hidden_layers)
    ] if mlps else []

    attns = [
        f"gpt_neox.layers.{layer}.attention.dense"
        for layer in range(config.num_hidden_layers)
    ] if attns else []

    return mlps + attns

def parse_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    def process_function(res):
        input_ids = tokenizer(res["prompt"])["input_ids"]
        label = tokenizer(res["label"])["input_ids"][-1]
        return dict(
            input_ids=input_ids,
            label=label,
        )

    return dataset.map(process_function)

def parse_test_dataset(
    dataset: Dataset,
    options: List[str],
    tokenizer,
) -> Tuple[Dataset, List[int]]:

    class_tokens = [
        tokenizer(options[i])["input_ids"][-1]
        for i in range(len(options))
    ]

    return parse_dataset(dataset, tokenizer), class_tokens

def eval_model_acc_peft(
    model: torch.nn.Module,
    ds: Dataset,
    class_tokens: List[int],
    eval_batch_size: int = 16,
) -> None:
    """
    This function evaluates the accuracy of a given model on a given dataset.

    Parameters:
    model (nn.Module): The model to be evaluated.
    ds (datasets.Dataset): The dataset on which the model is to be evaluated.

    Returns:
    results (list): A list of dictionaries containing the input_ids, ground truth label, predicted label,
                    accuracy of prediction, logits and soft label for each example in the dataset.
    """

    def unpack(x):
        assert isinstance(x, torch.Tensor), type(x)
        return x.detach().float().cpu().numpy().tolist()

    def to_batch(x, batch_size):
        for i in range(0, len(x), batch_size):
            yield x[i : i + batch_size]

    model.eval()

    with torch.no_grad():
        results = []
        # for ex in ds:
        for batch in to_batch(ds, eval_batch_size):
            # pad input_ids to common length
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ex) for ex in batch["input_ids"]], batch_first=True
            ).to(model.device if hasattr(model, "device") else "cpu")

            input_lens = (input_ids != 0).sum(dim=-1)

            labels = [class_tokens.index(ex) for ex in batch["label"]]

            # run forward pass
            raw_logits = model(input_ids).logits
            raw_logits = torch.stack([
                raw_logits[i, input_lens[i] - 1, class_tokens] for i in range(len(input_lens))
            ])

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            preds = raw_logits.argmax(dim=-1).detach().cpu().numpy().tolist()

            results.extend(
                [
                    dict(
                        pred=pred,
                        prob=prob,
                        label=label,
                    )
                    for pred, prob, label in zip(preds, probs, labels)
                ]
            )
        
        # calculate AUROC
        preds = np.array([result["pred"] for result in results])
        probs = np.array([result["prob"] for result in results])
        labels = np.array([result["label"] for result in results])

        auroc = roc_auc_score(labels, probs[:, 1])

        return auroc, preds

def replace_labels(
    ds: Dataset,
    new_labels: list[int],
    class_labels: list[str],
) -> Dataset:
    # cursed, but idk how to get SFTrainer to work with soft labels.
    new_labels_ds = Dataset.from_dict({
        "completion": [class_labels[i] for i in new_labels],
    })

    ds = ds.remove_columns(["completion", "label", "input_ids"])

    ds = concatenate_datasets([new_labels_ds, ds], axis=1)

    return ds

def parse_dataset(
    dataset: Dataset,
    tokenizer,
) -> Dataset:
    def process_function(res):
        input_ids = tokenizer(res["prompt"])["input_ids"]
        label = tokenizer(res["completion"])["input_ids"][-1]
        return dict(
            input_ids=input_ids,
            label=label,
        )

    return dataset.map(process_function)

def parse_test_dataset(
    dataset: Dataset,
    options: List[str],
    tokenizer,
) -> Tuple[Dataset, List[int]]:

    class_tokens = [
        tokenizer(options[i])["input_ids"][-1]
        for i in range(len(options))
    ]

    return parse_dataset(dataset, tokenizer), class_tokens

def train_model_peft(
    model_name: str,
    label: str,
    save_path: Optional[str] = None,
    *,
    train_ds: Dataset,
    eval_ds: Dataset,
    class_labels: list[str],
    inference_ds: Optional[Dataset] = None,
    per_gpu_batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 2,
    model_ckpt: Optional[int] = None,
    lora_rank: int = 0,
    lora_modules: Optional[list[str]] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def format_fn(ex):
        return [
            prompt + label
            for prompt, label in zip(ex["prompt"], ex["completion"])
        ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        revision = f"step{model_ckpt}" if model_ckpt is not None else "main",
    )

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            fp16=True,
            output_dir=get_model_path(model_name, label, save_path, model_ckpt=model_ckpt),
            gradient_accumulation_steps=1,
            learning_rate=lr,
            logging_steps=50,
            num_train_epochs=epochs,
            optim="adamw_torch",
            adam_beta2=0.95,
            per_device_train_batch_size=per_gpu_batch_size,
            remove_unused_columns=False,
            report_to="none",
            save_steps=4000,
            warmup_steps=100,
            lr_scheduler_type="linear",
        ),
        formatting_func=format_fn,
        data_collator=LastTokenOnlyDataCollator(tokenizer, mlm=False),
        peft_config=(
            LoraConfig(  # type: ignore
                r=lora_rank, target_modules=lora_modules
            )
            if lora_rank > 0
            else None
        ),
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    return trainer, model

def train_or_load_model_peft(
    model_name: str,
    label: str,
    save_path: Optional[str] = None,
    *,
    train_ds: Dataset,
    eval_ds: Dataset,
    class_labels: list[str],
    inference_ds: Optional[Dataset] = None,
    per_gpu_batch_size: int = 8,
    lr: float = 1e-4,
    epochs: int = 2,
    model_ckpt: Optional[int] = None,
    lora_rank: int = 0,
    lora_modules: Optional[list[str]] = None,
    force_retrain: bool = False,
):
    if model_ckpt is not None:
        print(f"Finetuning {model_name}/{label}, step = {model_ckpt}.")
    else:
        print(f"Finetuning {model_name}/{label}.")

    # check if model already exists
    path = get_model_path(model_name, label, save_path, model_ckpt=model_ckpt)

    if path is not None and os.path.exists(path) and not force_retrain:
        print("Model already exists, skipping training.")
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
        )
    else:
        trainer, model = train_model_peft(
            model_name,
            label,
            save_path=save_path,
            train_ds=train_ds,
            eval_ds=eval_ds,
            class_labels=class_labels,
            inference_ds=inference_ds,
            per_gpu_batch_size=per_gpu_batch_size,
            lr=lr,
            epochs=epochs,
            model_ckpt=model_ckpt,
            lora_rank=lora_rank,
            lora_modules=lora_modules,
        )
        if path is not None:
            trainer.save_model(path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_ds, class_tokens = parse_test_dataset(
        eval_ds,
        class_labels,
        tokenizer,
    )

    eval_auroc, _ = eval_model_acc_peft(
        model,
        eval_ds,
        class_tokens,
    )

    print(f"Eval AUROC: {eval_auroc}")

    new_ds = None

    if inference_ds is not None:
        inference_ds, class_tokens = parse_test_dataset(
            inference_ds,
            class_labels,
            tokenizer,
        )

        inference_auroc, inferred_labels = eval_model_acc_peft(
            model,
            inference_ds,
            class_tokens,
        )

        print(f"Inference AUROC: {inference_auroc}")

        new_ds = replace_labels(
            inference_ds,
            inferred_labels,
            class_labels,
        )
    
    return eval_auroc, new_ds

def delete_saved_model_peft(
    model_name: str,
    label: str,
    save_path: Optional[str] = None,
    model_ckpt: Optional[int] = None,
) -> None:
    path = get_model_path(model_name, label, save_path, model_ckpt=model_ckpt)
    if os.path.exists(path):
        shutil.rmtree(path)