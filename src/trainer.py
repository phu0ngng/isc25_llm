import os
import logging
import time
import glob
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Trainer, TrainingArguments
from .dataset import get_data_collator
from .evaluation import CausalLMEvaluator
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)


class CustomTrainer:
    def __init__(self, config, model, tokenizer, train_dataset, eval_dataset=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = get_data_collator(
            tokenizer=tokenizer, max_length=config.max_length
        )

    def train_speed(self, local_rank):
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.num_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim="adamw_torch_fused",
            learning_rate=self.config.learning_rate,
            fp16=self.config.precision == "fp16",
            bf16=self.config.precision == "bf16",
            ddp_find_unused_parameters=False,
            report_to="none",
            remove_unused_columns=False,
            torch_compile=False,
            # Checkpoint settings
            save_strategy="epoch",
            save_total_limit=2,
            logging_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            # Feel free to define your own compute_metrics when enabling the validation
            compute_metrics=None
        )

        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time

        if local_rank == 0:
            logger.info(f"Total training time: {total_time:.2f} seconds")

        self.save_checkpoint(self.config.num_epochs)

    def evaluate_model(self):
        """Evaluate model on test dataset"""
        self.model.eval()  # Set model to evaluation mode
        evaluator = CausalLMEvaluator(self.model, self.tokenizer)

        with torch.no_grad():  # Disable gradient calculation
            results = evaluator.evaluate(self.eval_dataset)

        return results

    def save_checkpoint(self, epoch):
        """Save checkpoint with distributed training in mind"""
        if not self.is_main_process():
            return None

        os.makedirs(self.config.checkpoint.dir, exist_ok=True)

        # Save model state
        checkpoint = {
            "epoch": epoch,
            # 'optimizer_state_dict': self.optimizer.state_dict(),
            # We need these two to verify your submission
            "lora_state_dict": get_peft_model_state_dict(self.model.module if isinstance(self.model, DDP) else self.model),            
            "config": self.config,
        }

        # Regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint.dir, "best_model_lora.pt"
        )
        logger.info("Checkpointing from local_rank = 0 ...")
        torch.save(checkpoint, checkpoint_path)

        # Remove old checkpoints if needed
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Keep only last K checkpoints"""
        if self.config.checkpoint.keep_last_k <= 0:
            return

        checkpoints = sorted(
            glob.glob(os.path.join(self.config.checkpoint.dir, "checkpoint_epoch_*.pt"))
        )

        for checkpoint in checkpoints[: -self.config.checkpoint.keep_last_k]:
            os.remove(checkpoint)

    @staticmethod
    def is_main_process():
        """Check if this is the main process (rank 0)"""
        return (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )
