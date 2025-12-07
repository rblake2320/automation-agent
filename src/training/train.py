"""
Training Script for VBScript/Selenium Automation Agent
Wrapper for Axolotl training with RTX 5090 optimizations
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

from loguru import logger


def check_environment():
    """Verify training environment is properly set up."""
    logger.info("Checking training environment...")

    # Check CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA is not available!")
            return False

        device_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"GPU: {device_name}")
        logger.info(f"VRAM: {vram:.1f} GB")

        if "5090" in device_name or vram >= 32:
            logger.info("RTX 5090 detected - using optimized settings")

    except ImportError:
        logger.error("PyTorch not installed!")
        return False

    # Check Axolotl
    try:
        import axolotl
        logger.info(f"Axolotl version: {axolotl.__version__}")
    except ImportError:
        logger.error("Axolotl not installed! Run: pip install axolotl")
        return False

    # Check flash-attention
    try:
        import flash_attn
        logger.info(f"Flash Attention available: {flash_attn.__version__}")
    except ImportError:
        logger.warning("Flash Attention not installed - training will be slower")

    # Check training data
    data_path = Path("./data/processed/training_data.jsonl")
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        logger.info("Run data processing first: python src/processors/data_processor.py")
        return False

    # Count examples
    with open(data_path, 'r') as f:
        num_examples = sum(1 for _ in f)
    logger.info(f"Training examples: {num_examples}")

    if num_examples < 1000:
        logger.warning("Low training data count. Consider collecting more data.")

    return True


def run_training(config_path: str = "configs/axolotl_config.yaml"):
    """Run the training using Axolotl."""

    if not check_environment():
        logger.error("Environment check failed. Fix issues before training.")
        return False

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    logger.info(f"Starting training with config: {config_path}")
    logger.info(f"Training started at: {datetime.now().isoformat()}")

    # Set environment variables for optimal RTX 5090 performance
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TOKENIZERS_PARALLELISM": "false",
    })

    # Run Axolotl training
    cmd = [
        sys.executable, "-m", "axolotl.cli.train",
        str(config_path)
    ]

    # Use accelerate for better multi-GPU support (even with single GPU)
    accelerate_cmd = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_processes", "1",
        "-m", "axolotl.cli.train",
        str(config_path)
    ]

    try:
        logger.info(f"Running: {' '.join(accelerate_cmd)}")

        process = subprocess.Popen(
            accelerate_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode == 0:
            logger.info("Training completed successfully!")
            logger.info(f"Training ended at: {datetime.now().isoformat()}")
            return True
        else:
            logger.error(f"Training failed with code: {process.returncode}")
            return False

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        process.terminate()
        return False

    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


def merge_lora(config_path: str = "configs/axolotl_config.yaml"):
    """Merge LoRA adapter with base model."""
    logger.info("Merging LoRA adapter with base model...")

    output_dir = Path("./outputs")
    checkpoints = list(output_dir.glob("checkpoint-*"))

    if not checkpoints:
        logger.error("No checkpoints found!")
        return False

    # Get latest checkpoint
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
    logger.info(f"Using checkpoint: {latest}")

    cmd = [
        sys.executable, "-m", "axolotl.cli.merge_lora",
        config_path,
        "--lora_model_dir", str(latest),
        "--output_dir", "./merged_model"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("LoRA merge completed!")
        logger.info("Merged model saved to: ./merged_model")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Merge failed: {e}")
        return False


def export_to_gguf():
    """Export model to GGUF format for Ollama."""
    logger.info("Exporting to GGUF format...")

    merged_model = Path("./merged_model")
    if not merged_model.exists():
        logger.error("Merged model not found. Run merge_lora first.")
        return False

    # Check if llama.cpp is available
    llama_cpp = Path("./llama.cpp")
    if not llama_cpp.exists():
        logger.info("Cloning llama.cpp...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)

    # Convert to GGUF
    convert_script = llama_cpp / "convert_hf_to_gguf.py"

    cmd = [
        sys.executable, str(convert_script),
        str(merged_model),
        "--outtype", "q4_k_m",
        "--outfile", "./automation-agent-q4_k_m.gguf"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("GGUF export completed!")
        logger.info("Model saved to: ./automation-agent-q4_k_m.gguf")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Export failed: {e}")
        return False


def create_ollama_model():
    """Create Ollama model from GGUF file."""
    logger.info("Creating Ollama model...")

    gguf_file = Path("./automation-agent-q4_k_m.gguf")
    if not gguf_file.exists():
        logger.error("GGUF file not found. Run export_to_gguf first.")
        return False

    # Create Modelfile
    modelfile_content = '''FROM ./automation-agent-q4_k_m.gguf

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
PARAMETER stop "</s>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''

    modelfile_path = Path("./Modelfile")
    modelfile_path.write_text(modelfile_content)
    logger.info("Created Modelfile")

    # Create Ollama model
    cmd = ["ollama", "create", "automation-agent", "-f", str(modelfile_path)]

    try:
        result = subprocess.run(cmd, check=True)
        logger.info("Ollama model created successfully!")
        logger.info("Run with: ollama run automation-agent")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ollama create failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("Ollama not found. Install from https://ollama.ai")
        return False


def main():
    """Main entry point with command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Train VBScript/Selenium Automation Agent")
    parser.add_argument("command", choices=["train", "merge", "export", "ollama", "all"],
                       help="Command to run")
    parser.add_argument("--config", default="configs/axolotl_config.yaml",
                       help="Path to Axolotl config")
    args = parser.parse_args()

    if args.command == "train":
        run_training(args.config)

    elif args.command == "merge":
        merge_lora(args.config)

    elif args.command == "export":
        export_to_gguf()

    elif args.command == "ollama":
        create_ollama_model()

    elif args.command == "all":
        # Run full pipeline
        if run_training(args.config):
            if merge_lora(args.config):
                if export_to_gguf():
                    create_ollama_model()


if __name__ == "__main__":
    main()
