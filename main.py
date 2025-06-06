import os
import copy
from dataclasses import asdict
import logging
from argparse import ArgumentParser
from src.lora_model import LoRAModel
from src.trainer import CustomTrainer
from src.distributed import DistributedSetup
from src.dataset import load_dataset_for_training
from src.hub_utils import HFSetup
from src.evaluation import CausalLMEvaluator
from config import TrainingConfig, HardwareConfig
from pprint import pformat


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)


def setup_cache_dir(config: TrainingConfig, rank: int):
    """Setup HuggingFace environment including cache and authentication"""
    # Setup cache directories (all ranks need this)
    cache_dirs = HFSetup.setup_cache_dirs(config.cache.base_cache_dir)
    logger.info("Cache directories configured at: %s", config.cache.base_cache_dir)

    # Only rank 0 needs to handle authentication
    if rank == 0 and config.cache.use_auth and config.cache.hf_token:
        success = HFSetup.login_huggingface(config.cache.hf_token)
        if not success:
            print("Warning: HuggingFace authentication failed, continuing without auth")


def main():
    parser = ArgumentParser(
        description="*** ISC25 SCC - Fine Tuning LLama with LoRA ***"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["speed", "accuracy"],
        required=True,
        help="Choose between speed or accuracy benchmark",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        choices=["cuda", "xpu", "hip", "cpu"],
        default="cuda",
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for training restart or evaluation",
    )
    args = parser.parse_args()

    # Load config
    hardware_config = HardwareConfig(
        device_type=args.device_type, precision=args.precision
    )
    # Setup distributed training if needed
    local_rank = DistributedSetup.setup_distributed(hardware_config)

    config = TrainingConfig()
    config.hardware = hardware_config

    # Mask the sensitive token in the copy
    config_copy = copy.deepcopy(config)
    config_copy.cache.hf_token = "hf_secret_token"
    logger.info("Training Configuration:\n%s", pformat(asdict(config_copy)))

    # Setup environment (all ranks need cache dirs, only rank 0 needs auth)
    setup_cache_dir(config, local_rank)

    # Ensure all processes are ready before model creation
    DistributedSetup.barrier()

    # Setup model and tokenizer
    model_handler = LoRAModel(config)
    if args.checkpoint:
        config.if_checkpoint = True
        checkpoint_path = args.checkpoint
        logger.info("Loading checkpoint from %s", args.checkpoint)
        if os.path.isdir(checkpoint_path):
            # If directory, load the best model
            checkpoint_path = os.path.join(checkpoint_path, "best_model_lora.pt")

        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        model, tokenizer = model_handler.load_checkpoint(checkpoint_path, local_rank)
    else:
        logger.info("Initializing new model")
        model, tokenizer = model_handler.setup_model(local_rank)

    DistributedSetup.barrier()

    # Wrap model with appropriate parallel strategy
    logger.info("Wrapping the model with parallelism")
    model = DistributedSetup.wrap_model_for_distributed(model, config, local_rank)

    DistributedSetup.barrier()

    run_training = args.benchmark == "speed" or (args.benchmark == "accuracy" and not args.checkpoint)
    run_evaluation = args.benchmark == "accuracy"

    if run_training:
        train_dataset = load_dataset_for_training(
            dataset_name=config.speed_dataset,
            tokenizer=tokenizer,
            prompt_config=config.prompt,
            split="all",
            max_length=config.max_length,
        )
    else:
        train_dataset = None

    if run_evaluation:
        eval_dataset = load_dataset_for_training(
            dataset_name=config.accuracy_dataset,
            tokenizer=tokenizer,
            prompt_config=config.prompt,
            split="test",
            max_length=config.max_length,
        )
    else:
        eval_dataset = None

    trainer = CustomTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    if run_training:
        logger.info("Training ...")
        trainer.train_speed(local_rank)

    if run_evaluation:
        # Initialize evaluator and run evaluation
        evaluator = CausalLMEvaluator(model, tokenizer)
        results = evaluator.evaluate(eval_dataset)

        if DistributedSetup.is_main_process(local_rank):
            logger.info("Evaluation Results:")
            logger.info("Accuracy: %.4f", results['accuracy'])
            logger.info("Correct Predictions: %d", results['correct_predictions'])
            logger.info("Total Samples: %d", results['total_samples'])

    # Cleanup
    DistributedSetup.cleanup_distributed()
    logger.info("DONE: local_rank = %d", local_rank)


if __name__ == "__main__":
    main()
