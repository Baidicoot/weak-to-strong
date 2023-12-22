from weak_to_strong.train_peft import (
    train_or_load_model_peft,
    delete_saved_model_peft,
    get_neox_lora_modules,
)

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoConfig

from random import Random
import functools
from typing import Optional, List, Tuple

from fire import Fire

import json
import os

ckpts = list(range(1000, 144000, 1000))

def process_dataset(ds_name: str, formatter, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    results = {}
    for split, n_docs in split_sizes.items():
        ds = load_dataset(ds_name, split=split)
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        ds = ds.map(functools.partial(formatter, rng=Random(seed)))
        ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
        results[split] = ds
    return results

def main(
    weak_model_name: str = "EleutherAI/pythia-410m-v0",
    strong_model_names: list[str] = [
        #"EleutherAI/pythia-410m-v0",
        "EleutherAI/pythia-1b-v0",
        "EleutherAI/pythia-1.4b-v0",
        "EleutherAI/pythia-2.8b-v0",
    ],
    valid_ckpts: list[int] = ckpts,
    n_docs: int = 10_000,
    n_eval: int = 1_000,
    save_path: str = "results",
    seed: int = 42,
):
    SCIQ_FORMAT = "Q: {question} Is the answer \"{answer}\"?\nA (True/False):"
    labels = (" False", " True")

    def format_sciq(ex, rng):
        true_answer = rng.random() < 0.5

        if true_answer:
            answer = ex["correct_answer"]
            label = labels[1]
        else:
            answer = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
            label = labels[0]

        prompt = SCIQ_FORMAT.format(
            question=ex["question"],
            answer=answer,
        )
        
        return dict(prompt=prompt, completion=label)

    dataset = process_dataset("sciq", format_sciq, seed=seed, split_sizes=dict(train=n_docs, test=n_eval))
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    train_dataset = train_dataset.train_test_split(test_size=0.5, seed=seed)
    finetune_dataset, inference_dataset = train_dataset["train"], train_dataset["test"]

    lora_modules = [
        "dense_h_to_4h",
        "dense_4h_to_h",
        "query_key_value",
    ]

    target_auroc, weak_ds = train_or_load_model_peft(
        weak_model_name,
        "weak_teacher",
        save_path=save_path,

        train_ds=finetune_dataset,
        eval_ds=eval_dataset,
        class_labels=labels,
        inference_ds=inference_dataset,
        lora_rank=8,
        lora_modules=lora_modules,
    )

    ckpt_aurocs = {}
    best_ckpts = {}
    best_aurocs = {}
    transfer_aurocs = {}

    results_path = f"{save_path}/results.json"

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            json.dump(dict(
                ckpt_aurocs=ckpt_aurocs,
                best_ckpts=best_ckpts,
                best_aurocs=best_aurocs,
                transfer_aurocs=transfer_aurocs,
            ), f)

    for strong_model_name in strong_model_names:
        with open("results.json", "r") as f:
            results = json.load(f)
            best_ckpts = results["best_ckpts"]
            if strong_model_name in best_ckpts:
                print(f"Skipping {strong_model_name}, already done.")
                continue

        short_name = strong_model_name.split("/")[-1]
        restore_prev_path = f"{save_path}/strong_gt/{short_name}_partial.json"
        if os.path.exists(restore_prev_path):
            with open(restore_prev_path, "r") as f:
                partial_results = json.load(f)
                ckpt_aurocs[strong_model_name] = partial_results["ckpt_aurocs"]
                endpoint_aurocs = partial_results["endpoint_aurocs"]
                interval = partial_results["interval"]
                print(f"Restored from previous run, searching: {interval}")
        else:
            ckpt_aurocs[strong_model_name] = {}

            interval = ckpts

            min_auroc, _ = train_or_load_model_peft(
                strong_model_name,
                "strong_gt",
                save_path=save_path,
                model_ckpt=interval[0],

                train_ds=finetune_dataset,
                eval_ds=eval_dataset,
                class_labels=labels,
                lora_rank=8,
                lora_modules=lora_modules,
                force_retrain=True,
            )

            max_auroc, _ = train_or_load_model_peft(
                strong_model_name,
                "strong_gt",
                save_path=save_path,
                model_ckpt=interval[-1],

                train_ds=finetune_dataset,
                eval_ds=eval_dataset,
                class_labels=labels,
                lora_rank=8,
                lora_modules=lora_modules,
                force_retrain=True,
            )

            endpoint_aurocs = [min_auroc, max_auroc]

            ckpt_aurocs[strong_model_name][interval[0]] = min_auroc
            ckpt_aurocs[strong_model_name][interval[-1]] = max_auroc

        while len(interval) > 2:
            with open(restore_prev_path, "w+") as f:
                json.dump(dict(
                    endpoint_aurocs=endpoint_aurocs,
                    ckpt_aurocs=ckpt_aurocs[strong_model_name],
                    interval=interval,
                ), f)

            midpoint = len(interval) // 2
            ckpt = interval[midpoint]

            midpoint_auroc, _ = train_or_load_model_peft(
                strong_model_name,
                "strong_gt",
                save_path=save_path,
                model_ckpt=ckpt,

                train_ds=finetune_dataset,
                eval_ds=eval_dataset,
                class_labels=labels,
                lora_rank=8,
                lora_modules=lora_modules,
                force_retrain=True,
            )

            ckpt_aurocs[strong_model_name][ckpt] = midpoint_auroc

            if midpoint_auroc < target_auroc:
                del_ckpt = interval[0]
                delete_saved_model_peft(strong_model_name, "strong_gt", save_path, del_ckpt)

                interval = interval[midpoint:]
                endpoint_aurocs[0] = midpoint_auroc
            else:
                del_ckpt = interval[-1]
                delete_saved_model_peft(strong_model_name, "strong_gt", save_path, del_ckpt)

                interval = interval[:midpoint+1]
                endpoint_aurocs[1] = midpoint_auroc

        if endpoint_aurocs[0] - target_auroc < target_auroc - endpoint_aurocs[1]:
            best_ckpt = interval[0]
            best_auroc = endpoint_aurocs[0]
        else:
            best_ckpt = interval[-1]
            best_auroc = endpoint_aurocs[1]
        
        best_ckpts[strong_model_name] = best_ckpt
        best_aurocs[strong_model_name] = best_auroc

        transfer_auroc, _ = train_or_load_model_peft(
            strong_model_name,
            "strong_transfer",
            save_path=save_path,
            model_ckpt=best_ckpt,

            train_ds=weak_ds,
            eval_ds=eval_dataset,
            class_labels=labels,
            lora_rank=8,
            lora_modules=lora_modules,
        )

        transfer_aurocs[strong_model_name] = transfer_auroc

        with open(results_path, "w") as f:
            json.dump(dict(
                ckpt_aurocs=ckpt_aurocs,
                best_ckpts=best_ckpts,
                best_aurocs=best_aurocs,
                transfer_aurocs=transfer_aurocs,
            ), f)
        
        os.remove(restore_prev_path)
        
        print("Results:")
        print("Teacher AUROC:", target_auroc)
        print("Ground-truth AUROC:", best_auroc)
        print("Transfer AUROC:", transfer_auroc)
    
    for strong_model_name in strong_model_names:
        print(f"{strong_model_name}:")
        print(f"  Teacher AUROC: {target_auroc}")
        print(f"  Ground-truth AUROC: {best_aurocs[strong_model_name]}")
        print(f"  Transfer AUROC: {transfer_aurocs[strong_model_name]}")

if __name__ == "__main__":
    Fire(main)