# ISC25 Student Cluster Competition: LLaMA Fine-Tuning Task

## Introduction

Welcome to the ISC25 Student Cluster Competition! This year's challenge focuses on optimizing the fine-tuning process of LLaMA 3.1 8B using LoRA (Low-Rank Adaptation). Teams will compete in both performance optimization and model accuracy enhancement.

## Competition Tasks
### 1. Speed Benchmark
- **Objective**: Achieve as high number of train sample per second as possible
- **Dataset**: CosmosQA will be given for practicing at home. A new train dataset will be given at the competition.
- **Measurement**: `train_samples_per_second`
- Teams have the freedom to fine-tune on the given dataset with any split.
- Teams have the freedom to run as many epochs as they want (one epoch at the least).

### 2. Accuracy Benchmark
- **Objective**: Achieve highest accuracy on evaluation dataset with the fine-tuned model from (1)
- **Evaluation Dataset**: ScienceQA will be given for practicing at home. A new evalulate dataset will be given at the competition.

### Evaluation Criteria
1. **Speed Benchmark (30%)**
   - Highest `train_samples_per_second` on the train dataset
   - Must provide reproducible results

2. **Accuracy Benchmark (50%)**
   - Highest accuracy on the evaluation dataset
   - Must provide checkpoint for verification

3. **Technical Report (20%)**
   - Summary of optimization techniques
   - Discussion of trade-offs and decisions
   - Max 500 words; a template is provided in the report.md

### Optimization Freedom
Teams are encouraged to explore various optimization techniques, including but not limited to:
- Model optimization (parallelism, quantization, etc).
- LoRA configurations (parameter tuning or any types of LoRA (qLoRA, DoRA, etc))
- Training strategies (learning rates, batch sizes)
- System optimizations (memory management, I/O optimization)
- Hardware-specific optimizations
- Using [TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html).
- Other PyTorch-based extensions.

The only constraints are:
- Must use PyTorch as the base framework
- Must maintain the provided code structure
- Must be reproducible on competition hardware
- Must maintain the same loggings and checkpointing

## Hardware Support

This implementation supports multiple hardware architectures:
- NVIDIA GPUs
- AMD GPUs
- Intel GPUs
- CPU-only systems

However, only NVIDIA GPUs were tested. Please reach out if you experience an issue on other
hardware.

## Container Setup

### NVIDIA GPUs
```bash
# Pull the container
docker pull nvcr.io/nvidia/pytorch:25.01-py3

# Run with NVIDIA Container Runtime
docker run --gpus all \
    --runtime=nvidia \
    -v ${PWD}:/workspace \
    nvcr.io/nvidia/pytorch:25.01-py3
```

### AMD GPUs
```bash
# Pull the container
docker pull rocm/pytorch:latest

# Run with ROCm support
docker run \
    --device=/dev/kfd \
    --device=/dev/dri \
    --security-opt seccomp=unconfined \
    --group-add video \
    -v ${PWD}:/workspace \
    rocm/pytorch:latest
```

### Intel GPUs
```bash
# Pull the container
docker pull intel/intel-extension-for-pytorch:latest

# Run with Intel GPU support
docker run -it \
    --device=/dev/dri \
    --group-add video \
    -v ${PWD}:/workspace \
    intel/intel-extension-for-pytorch:latest
```

### CPU-Only
```bash
# Pull the container
docker pull python:3.9-slim

# Run container
docker run -it \
    -v ${PWD}:/workspace \
    python:3.9-slim
```

## HuggingFace Registration
Please register an account at [HuggingFace](https://huggingface.co/) and [obtain a user access token](https://huggingface.co/docs/hub/en/security-tokens)

## Version Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+
- 40GB+ GPU memory per device (recommended)
- Hardware-specific libraries as needed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/phu0ngng/isc25_llm.git
cd isc25_llama_lora
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up HuggingFace authentication:
```bash
export HF_TOKEN="your_token_here"
```

4. Set up the cache directory for the model and datasets (Optional):
```bash
export HF_HOME=/path/to/huggingface/cache

```

## Running the Competition Tasks

### Time-to-Solution Benchmark
```bash
# For NVIDIA GPUs
torchrun --nproc_per_node=8 main.py --benchmark speed --device-type cuda

# For AMD GPUs
python -m torch.distributed.launch --nproc_per_node=8 main.py --benchmark speed --device-type rocm

# For Intel GPUs
python -m intel_extension_for_pytorch.cpu.launch --nproc_per_node=8 main.py --benchmark speed --device-type xpu

# For CPU-only
python main.py --benchmark time --device-type cpu --precision fp32
```
The sample `run.sh` script is provided.

### Accuracy Benchmark
```bash
# Fine-tuning and evaluation
torchrun --nproc_per_node=8 main.py --benchmark accuracy --device-type cuda

# Running evaluation from checkpoint
torchrun --nproc_per_node=8 main.py --benchmark accuracy --device-type cuda --checkpoint path-to-checkpoint

# Same for other GPUs
```

## Configuration

All configuration parameters can be found in `config.py`. Key parameters include:
- Model name and configuration
- Dataset paths
- Training hyperparameters
- Hardware-specific settings
- LoRA configuration.

## Datasets
For practicing at home, two datasets will be given:
1. For Fine-Tuning - CosmosQA: The Cosmos QA dataset is a large-scale collection of 35,600 problems designed to test commonsense-based reading comprehension. It presents multiple-choice questions that require interpreting the likely causes and effects of events in everyday narratives, often necessitating reasoning beyond the explicit text content.

2. Accuracy Benchmark - ScienceQA: The ScienceQA dataset consists of approximately 21,208 multimodal multiple-choice science questions, covering various topics across natural, language, and social sciences. It includes both image and text contexts, with detailed annotations to support understanding the reasoning behind answers. This structure makes it a valuable tool for assessing and improving AI's reasoning capabilities. For the scope of the competition, all the images-based questions are removed thus only text-based questions are used.

New datasets will be given at the competition.

## Directory Structure
This repository provides a reference implementation for fine-tuning LLaMA 3.1 8B with LoRA using
PyTorch.
```
isc25_llm/
├── report.md           # Template report
├── config.py           # Configuration parameters
├── main.py             # Main training script
├── requirements.txt    # Dependencies
├── run.sh              # Runner script
└── src/                # Source code
    ├── dataset.py      # Data loading
    ├── distributed.py  # Distributed utilities
    ├── evaluation.py   # Model evaluation
    ├── hub_utils.py    # HuggingFace utilities
    ├── lora_model.py   # LoRA implementation
    └── trainer.py      # Trainer
```
## Submission Instructions

Teams must submit:
1. Final model checkpoint
2. Code modifications (if any)
3. Configuration files
4. Brief report detailing optimizations used

Please submit a zip file that includes all the files and directories following this hierarchy
```
isc25_llm/
├── report.md           # Template report
├── config.py           # Configuration parameters
├── main.py             # Main training script
├── requirements.txt    # Dependencies
├── run.sh              # Runner script
├── src/                # Source code
│   ├── dataset.py      # Data loading
│   ├── distributed.py  # Distributed utilities
│   ├── evaluation.py   # Model evaluation
│   ├── hub_utils.py    # HuggingFace utilities
│   ├── lora_model.py   # LoRA implementation
│   └── trainer.py      # Trainer
├── checkpoints/        # Checkpoint directory
│   ├── best_model.pt   # Your final model for evaluation
│   └── checkpoint_xx   # Other checkpoint states
└── loggings/           # Logging directory
    ├── speed.log       # Stdout when running the speed benchmark
    └── accuracy.log    # Stdout when running the accuracy benchmark
```
You should name your zip file as `isc25_llm_[team_name].tar.gz`.

```
tar -czvf isc25_llm_team_name.tar.gz /path/to/isc25_llm
```

## Additional Notes
- Monitor GPU memory usage
- Use appropriate precision for hardware
- Implement regular checkpointing
- Profile code for bottlenecks
- Test with smaller datasets first


For questions or issues, please contact the competition organizers.
Good luck, and may the best-optimized model win!
