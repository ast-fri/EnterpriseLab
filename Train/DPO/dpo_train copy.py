# ✅ Environment setup FIRST
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import json


# Config
base_model = "path/to/your/models/qwen3-8b"  # Your base model
data_path = "path/to/your/dpo_dataset.json"  # Your DPO dataset in the new format
output_dir = "./dpo_output/dpo_chkpt_v5"

print(output_dir)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ✅ Load model in bfloat16 (NO quantization for FSDP)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    # NO quantization_config
    # NO device_map (let FSDP handle it)
)


# Enable gradient checkpointing
model.gradient_checkpointing_enable()


# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj']
)


model = get_peft_model(model, peft_config)


# Load dataset
with open(data_path, "r") as f:
    dpo_data = json.load(f)
for item in dpo_data:
    item.pop("metadata", None)
dataset = Dataset.from_list(dpo_data)


print(f"Loaded {len(dataset)} DPO pairs")


# ✅ CHANGED: Calculate max_steps explicitly for FSDP
num_gpus = 4
per_device_batch = 1
grad_accum = 4
epochs = 2

# Steps per GPU per epoch
steps_per_epoch = len(dataset) // (num_gpus * per_device_batch * grad_accum)
max_steps = steps_per_epoch * epochs

print(f"\n=== TRAINING STEP CALCULATION ===")
print(f"Dataset size: {len(dataset)}")
print(f"Steps per epoch (per GPU): {steps_per_epoch}")
print(f"Total max_steps: {max_steps}")
print(f"===================================\n")


# Training config
training_args = DPOConfig(
    output_dir=output_dir,
    
    # Batch & epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=max_steps,  # ✅ CHANGED: Use max_steps instead of num_train_epochs
    
    # Learning rate
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    # warmup_ratio=0.1,
    warmup_steps=12,
    
    # DPO specific - Can handle longer context now
    beta=0.1,
    max_length=32768,             
    max_prompt_length=28000,      
    max_completion_length=4768,   
    truncation_mode="keep_start",
    
    # Optimization
    gradient_checkpointing=True,
    bf16=True,
    optim="adamw_torch",  # Changed from paged_adamw_8bit
    max_grad_norm=1.0,
    
    # Logging
    logging_steps=5,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=15,
    
    # Memory
    remove_unused_columns=False,
    dataloader_num_workers=0,
    report_to="none",
)


# Initialize trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)


# ✅ ADDED: Debug logging to verify configuration
print("\n=== TRAINER CONFIGURATION ===")
print(f"Actual train dataloader length: {len(dpo_trainer.get_train_dataloader())}")
print(f"Warmup steps: {dpo_trainer.args.warmup_steps}")
print(f"Initial learning rate: {training_args.learning_rate}")
print(f"LR scheduler: {training_args.lr_scheduler_type}")
print(f"==============================\n")


print("✅ Starting training...")
dpo_trainer.train()


print("\n✅ Training complete!")
dpo_trainer.save_model(f"{output_dir}/final_model")
print(f"✅ Model saved to: {output_dir}/final_model")
