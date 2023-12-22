# adapted from https://github.com/EleutherAI/elk-generalization/blob/main/elk_generalization/training/sft.py

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import os

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
    save_path: str,
    model_ckpt: Optional[int] = None,
) -> str:
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

def process_test_dataset(
    dataset: Dataset,
    tokenizer,
    max_ctx: int,
) -> Dataset:
    def process_function(res):
        input_ids = tokenizer(res["text"])["input_ids"]
        label = tokenizer(res["completion"])["input_ids"][-1]
        return dict(
            input_ids=input_ids,
            label=label,
        )

    return dataset.map(process_function)


def eval_model_acc_peft(
    model: torch.nn.Module,
    ds: Dataset,
    eval_batch_size: int = 16,
    options: Optional[List[int]] = None,
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

            labels = batch["label"]

            if options is not None:
                labels = [options.index(l) for l in labels]

            # run forward pass
            raw_logits = model(input_ids).logits
            raw_logits = torch.stack(
                [raw_logits[i, input_lens[i] - 1, :] for i in range(len(input_lens))]
            )

            if options is not None:
                raw_logits = raw_logits[:, options]

            probs = unpack(torch.nn.functional.softmax(raw_logits, dim=-1))
            preds = unpack(torch.argmax(raw_logits, axis=-1))

            results.extend(
                [
                    dict(
                        label=label,
                        pred=pred,
                        probs=prob,
                        acc=int(pred == label),
                    )
                    for label, pred, prob in zip(
                        labels, preds, probs
                    )
                ]
            )
        accs = [r["acc"] for r in results]
        print("Accuracy:", np.mean(accs), "+/-", np.std(accs) / np.sqrt(len(accs)))

        return Dataset.from_list(results)

def train_and_save_model_peft(
    model_name: str,
    train_ds: Dataset,
    test_ds: Dataset,
    per_gpu_batch_size: int = 8,
    lr: float = 2e-5,
    epochs: int = 2,
    model_ckpt: Optional[int] = None,
    lora_rank: int = 0,
    lora_modules: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    label: str = "default",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def format_fn(ex):
        return [
            text + completion
            for text, completion in zip(ex["text"], ex["completion"])
        ]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": torch.cuda.current_device()},
        #torch_dtype=torch.float32 if lora_rank <= 0 else "auto",
    )

    test_ds = process_test_dataset(test_ds, tokenizer, max_ctx=1024)

    eval_results = eval_model_acc_peft(
        model,
        test_ds,
        eval_batch_size=16,
    )
    print(eval_results)

    trainer = SFTTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=get_model_path(model_name, label, save_path, model_ckpt=model_ckpt),
            #fp16=True,
            gradient_accumulation_steps=1,
            learning_rate=lr,
            logging_steps=50,
            num_train_epochs=epochs,
            optim=("adamw_torch" if lora_rank > 0 else "adamw_bnb_8bit"),
            adam_beta2=0.95,
            per_device_train_batch_size=per_gpu_batch_size,
            remove_unused_columns=False,
            report_to="none",
            save_steps=4000,
            warmup_steps=1000,
            weight_decay=0.1,
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

    eval_results = eval_model_acc_peft(
        model,
        test_ds,
        eval_batch_size=16,
    )
    print(eval_results)

def delete_saved_peft_model(
    model_name: str,
    save_path: str,
    label: str,
    model_ckpt: Optional[int] = None,
) -> None:
    path = get_model_path(model_name, label, save_path, model_ckpt=model_ckpt)
    os.remove(path)

def eval_saved_peft_model(
    model_name: str,
    save_path: str,
    label: str,
    dataset: Dataset,
    model_ckpt: Optional[int] = None,
    eval_batch_size: int = 16,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(
        get_model_path(model_name, label, save_path, model_ckpt=model_ckpt),
        device_map="auto",
        torch_dtype=torch.float32,
    )

    return eval_model_acc_peft(model, dataset, eval_batch_size=eval_batch_size)