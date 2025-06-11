# src/data/dataset.py
from typing import Dict, List, Optional
import logging
import json
import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from config import PromptConfig


logger = logging.getLogger(__name__)


def get_data_dir():
    return os.environ.get('DATA_DIR', default='data')


class CausalLMMultipleChoiceDataset(Dataset):
    """
    A dataset class that handles multiple-choice datasets
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        prompt_config: PromptConfig,
        max_length: int = 512,
        split: str = "all",
        **kwargs,
    ):
        del kwargs
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_config = prompt_config
        self.train = split != "validation"

        # Load dataset
        if dataset_name in ["speed_dataset", "accuracy_dataset"]:
            data_dir = get_data_dir()
            if not data_dir:
                raise ValueError("data_dir must be provided")
            logging.info(f"Loading dataset from directory {data_dir}, ignoring the split={split}")
            data_path = os.path.join(data_dir, f"{dataset_name}.jsonl")
            self.dataset = []
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Preprocessed data not found at {data_path}.")
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.dataset.append(json.loads(line))
            logging.info(f"Loaded {len(self.dataset)} samples from {data_path}")
        elif dataset_name == "lmms-lab/ScienceQA":
            self.dataset = load_dataset(
                dataset_name, "ScienceQA-FULL", split=split, trust_remote_code=True
            )
            # Filter out questions with images
            original_length = len(self.dataset)
            self.dataset = self.dataset.filter(
                lambda example: example["image"] is None or example["image"] == ""
            )
            new_length = len(self.dataset)
            logging.info(
                f"Removed {original_length - new_length} examples with images. New dataset length: {new_length}"
            )
        else:
            self.dataset = load_dataset(
                dataset_name, split=split, trust_remote_code=True
            )

        # Set maximum lengths for different parts
        self.max_answer_length = 0
        self.max_prompt_length = max_length - self.max_answer_length

    def _format_prompt(self, example: Dict) -> str:
        if self.dataset_name in ["speed_dataset", "accuracy_dataset"]:
            context = example["context"]
            question = example["question"]
            choices = example["options"]
            label = example["answer"]

        elif self.dataset_name == "cosmos_qa":
            context = example["context"]
            question = example["question"]
            choices = [
                example["answer0"],
                example["answer1"],
                example["answer2"],
                example["answer3"],
            ]
            label = example["label"]

        else:  # science_qa
            context = example.get("lecture", "")
            question = example["question"]
            choices = example["choices"]
            label = example["answer"]

            if example.get("hint"):
                context = f"{context}\\nHint: {example['hint']}"

        # Format options
        options_text = "\\n".join(
            self.prompt_config.options_format.format(idx=i + 1, choice=choice)
            for i, choice in enumerate(choices)
        )

        # Format full prompt
        prompt = self.prompt_config.template.format(
            context=context, question=question, options=options_text
        )

        return prompt, label

    def __getitem__(self, idx: int) -> Dict:
        example = self.dataset[idx]
        prompt, label = self._format_prompt(example)

        # Tokenize prompt
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding="max_length",
            padding_side="left",
            truncation=True,
            return_tensors=None,  # Return python lists
        )

        # Convert label to answer string and tokenize
        answer = str(label + 1)  # Convert to 1-based index
        answer_encoding = self.tokenizer(
            answer,
            max_length=2,
            padding="max_length",
            padding_side="left",
            truncation=True,
            return_tensors=None,
        )

        input_ids = prompt_encoding["input_ids"]
        attention_mask = prompt_encoding["attention_mask"]
        if self.train:
            labels = [-100] * (self.max_length - 2) + answer_encoding["input_ids"]
        else:
            labels = answer_encoding["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.dataset)


class CausalLMDataCollator:
    """
    Data collator for causal language modeling.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict:
        # The .pad() method respects the tokenizer's padding_side setting.
        # This will use left-padding as configured in CausalLMMultipleChoiceDataset.
        # It also handles padding of labels with -100 by default.
        batch = self.tokenizer.pad(
            features,
            padding="longest",  # Pad to longest sequence in batch
            max_length=self.max_length,
            return_tensors="pt",
        )
        return batch


def load_dataset_for_training(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    prompt_config: PromptConfig,
    split: str = "all",
    max_length: int = 512,
    cache_dir: Optional[str] = None,
) -> CausalLMMultipleChoiceDataset:

    logger.info(f"Loading dataset {dataset_name} ({split} split)")

    dataset = CausalLMMultipleChoiceDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        prompt_config=prompt_config,
        max_length=max_length,
        split=split,
        cache_dir=cache_dir,
    )

    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def get_data_collator(
    tokenizer: PreTrainedTokenizer, max_length: int = 512
) -> CausalLMDataCollator:
    """
    Get the appropriate data collator
    """
    return CausalLMDataCollator(tokenizer, max_length)
