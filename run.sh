#!/bin/bash

#export HF_TOKEN="hf_abcxyz"
#export HF_HOME="/path/to/hf_cache"
export DATA_DIR="./data"

pip install -r requirements.txt

# Check if HF_TOKEN is set
if [ -z "${HF_TOKEN}" ]; then
	echo "Error: HF_TOKEN environment variable is not set."
	exit 1
fi

# Speed benchmark
# torchrun --nproc_per_node=8 main.py --benchmark speed --device-type cuda

# Speed benchmark and save stdout/loggings
torchrun --nproc_per_node=8 main.py --benchmark speed --device-type cuda > loggings/speed.log 2<&1


# Speed and accuracy benchmark
#torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda

# Accuracy benchmark
#torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints

# Accuracy benchmark and save stdout/loggings
torchrun --nproc_per_node=1 main.py --benchmark accuracy --device-type cuda --checkpoint checkpoints > loggings/accuracy.log 2<&1
