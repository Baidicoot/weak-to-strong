from weak_to_strong.train_peft import (
    train_and_save_model_peft,
    eval_saved_peft_model,
    delete_saved_peft_model,
    get_neox_lora_modules
)

from datasets import load_dataset

from random import Random
import functools
from typing import Optional

from fire import Fire

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
        "EleutherAI/pythia-1b-v0",
        "EleutherAI/pythia-1.4b-v0",
        "EleutherAI/pythia-2.8b-v0",
    ],
    valid_ckpts: list[int] = ckpts,
    n_docs: int = 10_000,
    n_eval: int = 1_000,
    save_path: str = "models",
    seed: int = 42,
):
    SCIQ_FORMAT = "Q: {question} Is the answer \"{answer}\"?\nA (Yes/No):"
    completions = (" No", " Yes")

    def format_sciq(ex, rng):
        true_answer = rng.random() < 0.5
        if true_answer:
            answer = ex["correct_answer"]
        else:
            answer = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
        
        completion = completions[int(true_answer)]

        text = SCIQ_FORMAT.format(
            question=ex["question"],
            answer=answer,
            true_answer=completion
        )
        
        return dict(text=text, completion=completion)

    dataset = process_dataset("sciq", format_sciq, seed=seed, split_sizes=dict(train=n_docs, test=n_eval))
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    train_dataset = train_dataset.train_test_split(test_size=0.5, seed=seed)
    finetune_dataset, inference_dataset = train_dataset["train"], train_dataset["test"]

    lora_modules = [
        "dense_h_to_4h",
        "dense_4h_to_h",
        "query_key_value",
    ]

    train_and_save_model_peft(
        weak_model_name,
        finetune_dataset,
        eval_dataset,
        lora_rank=8,
        lora_modules=lora_modules,
        save_path=save_path,
    )

if __name__ == "__main__":
    Fire(main)