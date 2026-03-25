# merge_sft_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
parser.add_argument('--base_model_path', required=True, help='Path to base model')
parser.add_argument('--adapter_path', required=True, help='Path to LoRA adapter')
parser.add_argument('--output_path', required=True, help='Output path for merged model')

args = parser.parse_args()

# Your paths
base_model_path = args.base_model_path
sft_adapter_path = args.adapter_path
merged_output_path = args.output_path

print("=" * 50)
print("Merging SFT LoRA adapters into base model...")
print("=" * 50)

# Step 1: Load base model
print("\n[1/5] Loading base Qwen3-8B model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("✓ Base model loaded")

# Step 2: Load your SFT adapters
print("\n[2/5] Loading your SFT LoRA adapters...")
model_with_adapters = PeftModel.from_pretrained(
    base_model, 
    sft_adapter_path,
    torch_dtype=torch.bfloat16
)
print("✓ SFT adapters loaded")

# Step 3: Merge adapters into base model
print("\n[3/5] Merging adapters into base model...")
merged_model = model_with_adapters.merge_and_unload()
print("✓ Adapters merged successfully")

# Step 4: Load tokenizer
print("\n[4/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
print("✓ Tokenizer loaded")

# Step 5: Save merged model
print(f"\n[5/5] Saving merged SFT model to {merged_output_path}...")
merged_model.save_pretrained(
    merged_output_path,
    safe_serialization=True  # Save as safetensors
)
tokenizer.save_pretrained(merged_output_path)
print("✓ Merged model saved")

print("\n" + "=" * 50)
print("SUCCESS! Your SFT model is ready to use.")
print("=" * 50)
print(f"\nMerged model location: {merged_output_path}")
print("\nNext steps:")
print("1. Serve with vLLM:")
print(f"   vllm serve {merged_output_path} --port 8000 --tensor-parallel-size 4")
print("\n2. Or use directly in your code with HuggingFace transformers")
