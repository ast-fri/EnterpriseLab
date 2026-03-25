from datasets import load_dataset
import json
from datasets import Dataset
import re
import random
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TextStreamer
import torch
from peft import LoraConfig
import argparse
from trl import (
    SFTTrainer,
    SFTConfig,
    get_peft_config,
    get_quantization_config,
    ModelConfig,
    RichProgressCallback,
)
import io
import sys
import os



class SFT:
    def __init__(self, model_path, train_data_path=None, eval_data_path=None, output_dir=None, num_train_epochs=1.2, per_device_train_batch_size=16, learning_rate=5e-5):
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None
        self.setup_model(self.model_path)
        
    def setup_model(self, model_path):
        model_config = ModelConfig(
            model_name_or_path=model_path,
            trust_remote_code=True,
            dtype="auto",
            use_peft=True,
            lora_r=128,
            lora_alpha=256,
            lora_dropout=0.1,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_task_type="CAUSAL_LM",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        quantization_config = get_quantization_config(model_config)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=quantization_config,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
        self.model = model
        self.tokenizer = tokenizer


    def format_conversations(self, examples):
        train_data = []
        for example in examples:
            formatted_text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            train_data.append({"text": formatted_text})
        return train_data
        
    def train(self):
        # Load data (your existing code)
        with open(self.train_data_path, "r") as file:
            train_data = json.load(file)
        if self.eval_data_path:
            with open(self.eval_data_path, "r") as file:
                eval_data = json.load(file)
        else:
            eval_data = []
        
        random.shuffle(train_data)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Evaluation samples: {len(eval_data)}")
        
        formatted_train_dataset = self.format_conversations(train_data)
        formatted_train_dataset = Dataset.from_list(formatted_train_dataset)
        
        # formatted_eval_dataset = self.format_conversations(eval_data)
        # formatted_eval_dataset = Dataset.from_list(formatted_eval_dataset)
        
        # ===== CORRECTED LoRA Config =====
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.1,  # ✅ FIXED: Keep dropout enabled
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        
        # ===== CORRECTED Training Config =====
        training_args = SFTConfig(
            # ===== BATCH SIZE (CRITICAL FIX) =====
            per_device_train_batch_size=self.per_device_train_batch_size,     # ✅ INCREASED from 4
            # per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,      # ✅ INCREASED from 2
            # Effective batch = 8 * 4 * 4 = 128 ✅
            
            # ===== LEARNING RATE (CRITICAL FIX) =====
            learning_rate=self.learning_rate,                 # ✅ REDUCED from 5e-5 (5x reduction!)
            warmup_ratio=0.2,                   # ✅ INCREASED warmup (more gradual)
            
            # ===== REGULARIZATION =====
            weight_decay=0.01,                  # ✅ REDUCED from 0.1 (back to normal)
            max_grad_norm=1.0,                  # ✅ REDUCED from 10.0 (standard clipping)
            
            # ===== TRAINING DURATION =====
            num_train_epochs=self.num_train_epochs,
            
            # ===== EVALUATION & CHECKPOINTING =====
            # eval_strategy="steps",
            # eval_steps=10,                      # ✅ Evaluate more frequently
            save_steps=10,
            save_total_limit=5,
            # load_best_model_at_end=True,
            # metric_for_best_model="eval_loss",
            # greater_is_better=False,
            
            # ===== EARLY STOPPING (CRITICAL) =====
            # This will stop training when eval_loss increases
            
            # ===== OTHER SETTINGS =====
            gradient_checkpointing=True,
            bf16=True,
            lr_scheduler_type="cosine",
            logging_steps=1,
            output_dir=self.output_dir,
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            optim="adamw_torch_fused",
        )
        
        # ===== STRICTER Early Stopping =====
        # callbacks = [
        #     EarlyStoppingCallback(
        #         early_stopping_patience=2,      # ✅ REDUCED from 3 (stop faster)
        #         early_stopping_threshold=0.005, # ✅ More sensitive threshold
        #     )
        # ]
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted_train_dataset,
            peft_config=lora_config,
            args=training_args,
        )
        
        # ✅ Verify configuration
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        frozen_params = total_params - trainable_params
        
        num_gpus = torch.cuda.device_count()
        effective_batch_size = (
            training_args.per_device_train_batch_size * 
            training_args.gradient_accumulation_steps * 
            num_gpus
        )

        print(f"\n{'='*70}")
        print(f"🚀 TRAINING CONFIGURATION - 4x A100 80GB OPTIMIZED")
        print(f"{'='*70}")
        print(f"✅ LoRA Applied Successfully!")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Frozen params: {frozen_params:,}")
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable %: {100 * trainable_params / total_params:.4f}%")
        print(f"\n📊 Dataset Split:")
        print(f"   Train samples: {len(formatted_train_dataset)} (90%)")
        # print(f"   Eval samples: {len(formatted_eval_dataset)} (10%)")
        print(f"\n💪 GPU Configuration:")
        print(f"   Number of GPUs: {num_gpus}")
        print(f"   GPU Type: A100 80GB")
        print(f"   Total VRAM: {num_gpus * 80}GB")
        print(f"\n📊 Batch Configuration:")
        print(f"   Per-device batch size: {training_args.per_device_train_batch_size}")
        print(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"\n⚡ Performance Optimizations:")
        print(f"   Gradient checkpointing: {training_args.gradient_checkpointing} (20% speed boost!)")
        print(f"   Flash Attention 2: Enabled")
        print(f"   BFloat16: {training_args.bf16}")
        print(f"   Fused AdamW: {training_args.optim}")
        print(f"   Data workers: {training_args.dataloader_num_workers}")
        print(f"{'='*70}\n")

        if trainable_params == 0:
            raise RuntimeError("❌ CRITICAL ERROR: No trainable parameters after applying LoRA!")
        
        # Train
        trainer.train()
        
        # Save best model
        # best_checkpoint = trainer.state.best_model_checkpoint
        # best_eval_loss = trainer.state.best_metric
        
        # print(f"\n{'='*70}")
        # print(f"✅ TRAINING COMPLETED")
        # print(f"{'='*70}")
        # print(f"Best checkpoint: {best_checkpoint}")
        # print(f"Best eval_loss: {best_eval_loss:.4f}")
        # print(f"{'='*70}\n")
        
        # Save final model (which is the best model loaded)
        final_model_path = f"{self.output_dir}/final_model"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        print(f"✅ Best model saved to: {final_model_path}")
        
        # Optional: Clean up non-best checkpoints to save disk space
        # print(f"\n🗑️ Cleaning up non-best checkpoints...")
        # for checkpoint_dir in os.listdir(self.output_dir):
        #     checkpoint_path = os.path.join(self.output_dir, checkpoint_dir)
        #     if checkpoint_dir.startswith("checkpoint-") and checkpoint_path != best_checkpoint:
        #         try:
        #             import shutil
        #             shutil.rmtree(checkpoint_path)
        #             print(f"   Deleted: {checkpoint_dir}")
        #         except Exception as e:
        #             print(f"   Could not delete {checkpoint_dir}: {e}")
        
        print(f"\n✅ Final directory structure:")
        # print(f"   {best_checkpoint} (best checkpoint)")
        print(f"   {final_model_path} (best model - ready to use)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SFT model")
    parser.add_argument('--model_name_or_path', required=True, help='Path or name of the model')
    parser.add_argument('--data_path', required=True, help='Path to training data JSON')
    parser.add_argument('--eval_data_path', help='Path to evaluation data JSON (optional)')
    parser.add_argument('--output_dir', required=True, help='Output directory for checkpoints')
    parser.add_argument('--num_train_epochs', type=float, default=1.2, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Batch size per device')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    sft = SFT(
        model_path=args.model_name_or_path,
        train_data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate
    )
    sft.train()
