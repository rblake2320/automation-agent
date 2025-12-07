"""
Full-precision LoRA Training Script for Automation Agent
Trains in fp16/bf16 WITHOUT 4-bit quantization - optimized for RTX 5090 32GB
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from loguru import logger


@dataclass
class TrainingConfig:
    """Training configuration for automation agent."""
    # Model settings
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model_path: Optional[str] = None

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training settings - conservative for fp16
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Memory optimizations
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Data settings
    max_seq_length: int = 1024
    train_data_path: str = "./data/processed/training_data.jsonl"

    # Logging
    logging_steps: int = 5
    save_steps: int = 50


def format_alpaca_prompt(example: dict) -> str:
    """Format example in Alpaca-style instruction format."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return prompt


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    prompts = [format_alpaca_prompt({
        "instruction": inst,
        "input": inp,
        "output": out
    }) for inst, inp, out in zip(
        examples["instruction"],
        examples.get("input", [""] * len(examples["instruction"])),
        examples["output"]
    )]

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_model_and_tokenizer(config: TrainingConfig):
    """Load model in fp16 precision with LoRA."""
    logger.info(f"Loading model: {config.model_name}")

    # Check for local model
    model_path = config.model_path or config.model_name
    local_model = Path("./models/deepseek-coder-6.7b-instruct")
    if local_model.exists():
        model_path = str(local_model)
        logger.info(f"Using local model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in fp16 (no quantization)
    logger.info("Loading model in bf16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Skip flash attention on Windows
    )

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def load_training_data(config: TrainingConfig, tokenizer):
    """Load and prepare training data."""
    logger.info(f"Loading training data from: {config.train_data_path}")

    dataset = load_dataset("json", data_files={"train": config.train_data_path})

    def tokenize_batch(examples):
        return tokenize_function(examples, tokenizer, config.max_seq_length)

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    logger.info(f"Training examples: {len(tokenized_dataset['train'])}")
    return tokenized_dataset


def train(config: TrainingConfig):
    """Run LoRA fine-tuning in fp16."""
    logger.info("=" * 60)
    logger.info("Starting FP16 LoRA Training (No Quantization)")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False

    device = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {device}")
    logger.info(f"VRAM: {vram:.1f} GB")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Load data
    dataset = load_training_data(config, tokenizer)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="adamw_torch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
    )

    # Train!
    logger.info("Starting training...")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Epochs: {config.num_train_epochs}")

    try:
        trainer.train()

        # Save final model
        final_path = Path(config.output_dir) / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        logger.info(f"Training complete!")
        logger.info(f"Model saved to: {final_path}")
        logger.info(f"Time: {datetime.now().isoformat()}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def merge_and_save(config: TrainingConfig, output_path: str = "./merged_model"):
    """Merge LoRA adapter with base model."""
    from peft import PeftModel

    logger.info("Merging LoRA adapter with base model...")

    final_path = Path(config.output_dir) / "final"
    if not final_path.exists():
        checkpoints = list(Path(config.output_dir).glob("checkpoint-*"))
        if not checkpoints:
            logger.error("No trained model found!")
            return False
        final_path = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))

    logger.info(f"Loading adapter from: {final_path}")

    # Load base model
    model_path = config.model_path or config.model_name
    local_model = Path("./models/deepseek-coder-6.7b-instruct")
    if local_model.exists():
        model_path = str(local_model)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, str(final_path))

    # Merge
    logger.info("Merging weights...")
    model = model.merge_and_unload()

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("Merge complete!")
    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="FP16 LoRA Training for Automation Agent")
    parser.add_argument("command", choices=["train", "merge", "all"],
                       help="Command to run")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    args = parser.parse_args()

    config = TrainingConfig(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    if args.command == "train":
        train(config)
    elif args.command == "merge":
        merge_and_save(config)
    elif args.command == "all":
        if train(config):
            merge_and_save(config)


if __name__ == "__main__":
    main()
