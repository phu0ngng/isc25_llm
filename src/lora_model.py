import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.distributed import DistributedSetup


class LoRAModel:
    def __init__(self, config):
        self.config = config
        self.hw_config = config.hardware

        # Set appropriate dtype based on precision and hardware
        if self.hw_config.precision == "bf16" and self.hw_config.device_type != "cpu":
            self.dtype = torch.bfloat16
        elif self.hw_config.precision == "fp16" and self.hw_config.device_type != "cpu":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def setup_model(self, local_rank, with_lora_init=True):
        # Ensure all processes use the same initialization
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Set up tokenizer with proper padding
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # Set padding token to be the EOS token

        # Set padding side to left
        tokenizer.padding_side = "left"

        # Set deterministic initialization
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=self.dtype, device_map=None
        )

        # Make sure model knows about padding token
        model.config.pad_token_id = tokenizer.pad_token_id

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if with_lora_init:
            lora_config = LoraConfig(
                r=self.config.lora.rank,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.lora.target_modules,
                init_lora_weights=True,
            )

            model = get_peft_model(model, lora_config)

        # Synchronize after LoRA application
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return model, tokenizer

    def load_checkpoint(self, checkpoint_path, local_rank):
        """Load checkpoint with both model and optimizer states"""
        base_model, tokenizer = self.setup_model(local_rank, with_lora_init=False)

        # Load via PEFT's method
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=True
        )

        # Move model to appropriate device
        model = model.to(DistributedSetup.get_device(self.hw_config, local_rank))

        return model, tokenizer
