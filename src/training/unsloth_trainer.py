"""
Unsloth Training Script for Automation Agent
Optimized for RTX 5090 32GB - 2-5x faster than standard PEFT
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from loguru import logger

# Must import unsloth before transformers
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq


# Configuration
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA config
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training config
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1


def format_prompt(example):
    """Format training example as instruction prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
{output}"""


def load_training_data(data_path: str):
    """Load and prepare training data."""
    logger.info(f"Loading data from: {data_path}")

    dataset = load_dataset("json", data_files={"train": data_path})

    # Format prompts
    def format_example(example):
        example["text"] = format_prompt(example)
        return example

    dataset = dataset.map(format_example)

    logger.info(f"Loaded {len(dataset['train'])} examples")
    return dataset["train"]


def train():
    """Run training with Unsloth."""
    logger.info("=" * 60)
    logger.info("Starting Unsloth Training for Automation Agent")
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

    # Check for local model
    model_path = MODEL_NAME
    local_model = Path("./models/deepseek-coder-6.7b-instruct")
    if local_model.exists():
        model_path = str(local_model)
        logger.info(f"Using local model: {model_path}")

    # Load model with Unsloth
    logger.info("Loading model with Unsloth (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=torch.bfloat16,
    )

    # Apply LoRA
    logger.info("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load data
    train_dataset = load_training_data("./data/processed/training_data.jsonl")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    # Train
    logger.info("Starting training...")
    try:
        trainer.train()

        # Save
        final_path = Path("./outputs/final")
        final_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(str(final_path))
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


def merge_and_save(output_path: str = "./merged_model"):
    """Merge LoRA and save full model."""
    logger.info("Merging LoRA adapter...")

    # Load trained model
    final_path = Path("./outputs/final")
    if not final_path.exists():
        logger.error("No trained model found!")
        return False

    # Load with unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(final_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # Load in full precision for merging
        dtype=torch.float16,
    )

    # Merge
    model = model.merge_and_unload()

    # Save
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"Merged model saved to: {output_path}")
    return True


def export_gguf(output_path: str = "./merged_model"):
    """Export to GGUF for Ollama."""
    logger.info("Exporting to GGUF...")

    merged_path = Path(output_path)
    if not merged_path.exists():
        logger.error("Merged model not found!")
        return False

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.float16,
    )

    # Save as GGUF
    try:
        model.save_pretrained_gguf(
            "automation-agent",
            tokenizer,
            quantization_method="q4_k_m"
        )
        logger.info("GGUF export complete!")
        return True
    except Exception as e:
        logger.error(f"GGUF export failed: {e}")
        return False


def create_ollama_model():
    """Create Ollama model from GGUF."""
    import subprocess

    logger.info("Creating Ollama model...")

    # Find GGUF file
    gguf_files = list(Path(".").glob("automation-agent*.gguf"))
    if not gguf_files:
        logger.error("No GGUF file found!")
        return False

    gguf_file = gguf_files[0]
    logger.info(f"Using GGUF: {gguf_file}")

    # Create Modelfile
    modelfile_content = f'''FROM ./{gguf_file.name}

SYSTEM """You are an expert automation engineer specializing in:
- VBScript for UFT/QTP (Micro Focus Unified Functional Testing)
- Selenium WebDriver (Python, Java, JavaScript)

Generate clean, production-ready automation code. Always include:
- Proper error handling (On Error Resume Next / try-catch)
- Descriptive variable names
- Comments for complex logic
- Best practices for test automation

For UFT VBScript:
- Use descriptive programming when appropriate
- Include Reporter.ReportEvent for logging
- Handle object synchronization with Wait/Exist
- Use DataTable for data-driven testing

For Selenium:
- Implement Page Object Model
- Use explicit waits (WebDriverWait)
- Handle dynamic elements properly
- Include proper assertions
"""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
'''

    Path("Modelfile").write_text(modelfile_content)

    # Create Ollama model
    try:
        subprocess.run(["ollama", "create", "automation-agent", "-f", "Modelfile"], check=True)
        logger.info("Ollama model created!")
        logger.info("Run with: ollama run automation-agent")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama create failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("Ollama not found!")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Unsloth Training for Automation Agent")
    parser.add_argument("command", choices=["train", "merge", "export", "ollama", "all"],
                       help="Command to run")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "merge":
        merge_and_save()
    elif args.command == "export":
        export_gguf()
    elif args.command == "ollama":
        create_ollama_model()
    elif args.command == "all":
        if train():
            if merge_and_save():
                if export_gguf():
                    create_ollama_model()


if __name__ == "__main__":
    main()
