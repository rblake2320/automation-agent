# Automation Agent - VBScript & Selenium LLM

A fine-tuned LLM specialized in generating VBScript (UFT/QTP) and Selenium automation code.

## Overview

This project creates a domain-specific language model optimized for:
- **VBScript/UFT/QTP**: Micro Focus Unified Functional Testing scripts
- **Selenium WebDriver**: Python, Java, and JavaScript automation

Built on `deepseek-coder-6.7b-instruct` with LoRA fine-tuning for RTX 5090 (32GB VRAM).

## Architecture

```
Data Collection → Data Processing → Training → Deployment
     ↓                  ↓              ↓           ↓
GitHub/SO          Alpaca Format   LoRA/Axolotl   Ollama
```

## Project Structure

```
automation-agent/
├── configs/
│   └── axolotl_config.yaml    # Training configuration
├── src/
│   ├── scrapers/
│   │   ├── github_scraper.py      # GitHub code collection
│   │   └── stackoverflow_scraper.py # Q&A collection
│   ├── processors/
│   │   ├── data_processor.py      # Data cleaning & formatting
│   │   └── uft_processor.py       # UFT script processing
│   └── training/
│       └── train.py               # Training wrapper
├── data/                          # Collected data (gitignored)
├── models/                        # Downloaded base model (gitignored)
├── outputs/                       # Training checkpoints (gitignored)
├── requirements.txt
├── run_pipeline.bat               # Full pipeline script
└── .env.example                   # Environment template
```

## Quick Start

### Prerequisites

- NVIDIA GPU with 24GB+ VRAM (RTX 4090/5090 recommended)
- Python 3.11+
- CUDA 12.4+
- Ollama (for deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/rblake2320/automation-agent.git
cd automation-agent

# Create environment
conda create -n automation-agent python=3.11 -y
conda activate automation-agent

# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Full Pipeline

```bash
# Windows
run_pipeline.bat

# Or step by step:
python src/scrapers/github_scraper.py --max-repos 30
python src/scrapers/stackoverflow_scraper.py --max-per-category 300
python src/processors/data_processor.py
python src/training/train.py train
python src/training/train.py merge
python src/training/train.py export
python src/training/train.py ollama
```

### Usage

```bash
# After training and deployment
ollama run automation-agent

# Example prompts:
# "Write a UFT VBScript to login to a web application"
# "Create a Selenium Python script with Page Object Model"
# "Generate error handling for UFT DataTable operations"
```

## Training Configuration

Optimized for RTX 5090 (32GB VRAM):

| Parameter | Value |
|-----------|-------|
| Base Model | deepseek-coder-6.7b-instruct |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Micro Batch | 4 |
| Gradient Accum | 4 |
| Learning Rate | 2e-4 |
| Epochs | 3 |
| Precision | bf16 |

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3090 24GB | RTX 5090 32GB |
| RAM | 32GB | 64GB+ |
| Storage | 100GB SSD | 500GB NVMe |
| CUDA | 12.1 | 12.8 |

## License

MIT License
