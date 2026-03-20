import os
import time
import json
import re
import asyncio
from typing import List, Dict
from torch.utils.data import Dataset
from trl import GRPOTrainer, GRPOConfig, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.cuda.amp import autocast
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()

# ---- GPTCaller Class ----
class GPTCaller:
    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        api_version: str = "2024-10-21",
        model_name: str = "gpt-4o",
        max_retries: int = 5
    ):
        self.api_key = api_key or os.getenv("AZURE_CHAT_API_KEY")
        self.api_base = api_base or os.getenv("AZURE_CHAT_ENDPOINT")
        self.api_version = api_version
        self.model_name = model_name
        self.max_retries = max_retries

        if not self.api_key or not self.api_base:
            raise ValueError("Azure API key and endpoint must be set")

        self.llm = AzureChatOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.api_base,
            model_name=self.model_name,
            temperature=0.3
        )

    async def __call__(
        self,
        prompt: str,
        response_format: str = "json",
        temperature: float = 0.3,
        max_tokens: int = 16384
    ) -> Dict:
        retries = 0
        last_error = None
        while retries < self.max_retries:
            try:
                if response_format == "json":
                    llm = self.llm.bind(
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    system_msg = "You are a helpful AI assistant. Always respond in valid JSON format."
                else:
                    llm = AzureChatOpenAI(
                        api_key=self.api_key,
                        api_version=self.api_version,
                        azure_endpoint=self.api_base,
                        model_name=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    system_msg = "You are a helpful AI assistant."

                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt)
                ]

                response = llm.invoke(messages)
                content = response.content

                if response_format == "json":
                    return json.loads(content)
                else:
                    return {"response": content}

            except Exception as e:
                last_error = e
                retries += 1
                time.sleep(15 * retries)
        return {} if response_format == "json" else {"response": "", "error": str(last_error)}


# ---- LLMJudge Class ----
class LLMJudge:
    def __init__(self, gpt_caller: GPTCaller):
        self.gpt_caller = gpt_caller

    def create_judge_prompt(self, query: str, generated_response: str, ground_truth: List[Dict]) -> str:
        """Create judge prompt with proper context"""
        # Extract expected answer from ground truth
        expected_answer = ""
        for msg in ground_truth:
            if msg.get('role') == 'assistant':
                expected_answer = msg.get('content', '')
                break
        
        return f"""You are an expert judge evaluating an AI agent's response. Score the response on multiple criteria.

        **Task:** {query}

        **Agent's Response:**
        {generated_response}

        **Expected Answer:** {expected_answer}

        Evaluate the response on these criteria and provide scores (0.0 to 1.0):

        1. **Answer Correctness** (0.0-2.0): Does the final answer match the expected answer?
        - 2.0 if exact match
        - 1.0 if partially correct or reasonable alternative
        - 0.0 if incorrect

        2. **Reasoning Quality** (0.0-1.0): Is the reasoning coherent and well-structured?

        3. **Tool Usage** (0.0-1.0): Are tools used appropriately and effectively?

        4. **Format Adherence** (0.0-0.5): Does the response follow required formatting?

        Return ONLY a JSON object with scores and a brief explanation:
        {{"answer_correctness": 0.0, "reasoning_quality": 0.0, "tool_usage": 0.0, "format_adherence": 0.0, "justification": "..."}}
        """

    async def judge_single_sample(self, sample: Dict) -> float:
        """Judge a single sample with proper field extraction"""
        query = sample.get("query", "")
        generated_text = sample.get("generated_text", "")
        ground_truth = sample.get("ground_truth", [])
        
        # DEBUG
        print(f"\nJudging sample:")
        print(f"  Query: {query[:100]}...")
        print(f"  Generated: {generated_text[:200]}...")
        print(f"  Ground truth available: {len(ground_truth) > 0}")
        
        prompt = self.create_judge_prompt(query, generated_text, ground_truth)
        result = await self.gpt_caller(
            prompt,
            response_format="json",
            temperature=0.0,
            max_tokens=512
        )
        
        # DEBUG
        print(f"  Judge result: {result}")
        
        if not result:
            reward = self._fallback_reward(generated_text, ground_truth)
            print(f"  Using fallback reward: {reward}")
            return reward
            
        reward = self.compute_weighted_reward(result, sample)
        print(f"  Computed reward: {reward}")
        return reward

    def compute_weighted_reward(self, eval_result: Dict, sample: Dict) -> float:
        try:
            ac = float(eval_result.get("answer_correctness", 0.0))
            rq = float(eval_result.get("reasoning_quality", 0.0))
            tu = float(eval_result.get("tool_usage", 0.0))
            fa = float(eval_result.get("format_adherence", 0.0))
            total = ac + 0.5 * (rq + tu) + fa
            sample["judge_feedback"] = eval_result.get("justification", "")
            return total
        except Exception as e:
            print(f"Error computing reward: {e}")
            return self._fallback_reward(sample.get("generated_text", ""), sample.get("ground_truth", []))

    def _fallback_reward(self, generated_text: str, ground_truth: List[Dict]) -> float:
        """Fallback reward computation"""
        reward = 0.0
        
        # Check for structured format
        try:
            if generated_text.strip().startswith('{'):
                parsed = json.loads(generated_text)
                if "thought" in parsed:
                    reward += 0.3
                if "tool" in parsed:
                    reward += 0.3
                if "final_answer" in parsed:
                    reward += 0.4
        except:
            # Fallback to simple pattern matching
            reward += 0.25 if "thought" in generated_text.lower() else 0.0
            reward += 0.25 if "tool" in generated_text.lower() else 0.0
        
        return reward

    async def batch_judge(self, samples: List[Dict]) -> List[float]:
        """Judge batch of samples"""
        tasks = [self.judge_single_sample(s) for s in samples]
        return await asyncio.gather(*tasks)


# ---- GRPODataset Class ----
class GRPODataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.system_prompt = """You are an Enterprise agent.
# Instructions:
- You always respond with a JSON object
- Generate a thought based on the user query
- Based on the thought select the most appropriate tool call from the list of available tools and output the tool_name
- Provide the tool_arguments of the tool extracted from the user query
- Strictly output the thought, tool and final_answer if any in the provided DICT Format only
- If you think you have the answer to the user query, or the task is executed, provide final answer, else keep final answer empty
# Output Format:
PROVIDE ONLY THE THOUGHT AND TOOLS, NOTHING ELSE
{{
    "thought": ......,
    "tool": {{
        "tool_name": ......,
        "tool_arguments": .......
    }},
    "final_answer": ....
}}
"""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item.get('messages', [])
        
        # Build prompt messages
        prompt_messages = []
        query = ""
        ground_truth = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                modified_msg = msg.copy()
                modified_msg['content'] = self.system_prompt + "\n\n" + msg.get('content', '')
                prompt_messages.append(modified_msg)
            elif msg.get('role') == 'user':
                query = msg.get("content", "")
                prompt_messages.append(msg)
            else:
                ground_truth.append(msg)
        
        # Format prompt for the model
        # Convert messages to text format
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {
            'prompt': prompt_text,  # ✅ GRPOTrainer expects 'prompt' field (not 'query')
            'ground_truth': ground_truth,
            'original_query': query,
        }


# ---- Reward Function Wrapper ----
def create_reward_function(judge: LLMJudge):
    """
    Create a reward function compatible with GRPOTrainer.
    
    With remove_unused_columns=False, all dataset columns are passed as kwargs.
    """
    def reward_fn(prompts, completions, completion_ids=None, **kwargs):
        """
        Compute rewards for generated completions.
        
        Args:
            prompts: Input prompts (list of strings)
            completions: Generated text completions (list of strings)
            completion_ids: Token IDs of completions (optional, not used)
            **kwargs: Dataset columns including 'ground_truth', 'original_query'
        
        Returns:
            List of reward scores (floats)
        """
        # DEBUG: Print to understand what we're receiving
        print(f"\n{'='*60}")
        print(f"REWARD FUNCTION CALLED")
        print(f"Number of prompts: {len(prompts)}")
        print(f"Number of completions: {len(completions)}")
        print(f"Available kwargs keys: {kwargs.keys()}")
        print(f"Sample prompt (first 200 chars): {prompts[0][:200] if prompts else 'NONE'}")
        print(f"Sample completion (first 500 chars): {completions[0][:500] if completions else 'NONE'}")
        
        # Extract dataset columns from kwargs
        ground_truths = kwargs.get('ground_truth', [[] for _ in range(len(completions))])
        original_queries = kwargs.get('original_query', prompts)  # Fallback to prompts
        
        print(f"Ground truth available: {len(ground_truths) > 0 and len(ground_truths[0]) > 0 if ground_truths else False}")
        print(f"Original queries available: {original_queries is not None}")
        print(f"{'='*60}\n")
        
        # Prepare samples for judging
        judge_samples = []
        for i, completion in enumerate(completions):
            ground_truth = ground_truths[i] if i < len(ground_truths) else []
            query = original_queries[i] if isinstance(original_queries, list) and i < len(original_queries) else prompts[i]
            
            judge_sample = {
                'query': query,
                'generated_text': completion,
                'ground_truth': ground_truth,
            }
            judge_samples.append(judge_sample)
        
        # Run async batch judging
        print(f"Calling judge with {len(judge_samples)} samples...")
        rewards = asyncio.run(judge.batch_judge(judge_samples))
        
        # DEBUG: Print rewards
        print(f"\n{'='*60}")
        print(f"REWARDS COMPUTED")
        print(f"Rewards: {rewards}")
        print(f"Mean reward: {sum(rewards) / len(rewards) if rewards else 0.0}")
        print(f"Min reward: {min(rewards) if rewards else 0.0}")
        print(f"Max reward: {max(rewards) if rewards else 0.0}")
        print(f"{'='*60}\n")
        
        return rewards
    
    return reward_fn


# ---- Main Training Script ----
if __name__ == "__main__":
    # Initialize GPT caller and judge
    gpt_caller = GPTCaller()
    judge = LLMJudge(gpt_caller=gpt_caller)
    
    # Model configuration
    model_name = "path/to/your/base/model"  # Replace with your model path or name
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model configuration
    model_config = ModelConfig(
        model_name_or_path=model_name,
        trust_remote_code=True,
        dtype="auto",
        use_peft=True,
        lora_r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_task_type="CAUSAL_LM",
    )

    # Load base model with BFloat16 (must match training config)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     dtype=torch.bfloat16,  # ✅ Match bf16=True in training config
    #     trust_remote_code=True,
    #     use_cache=False,
    #     low_cpu_mem_usage=True,
    #     device_map=None  # ✅ Let FSDP handle device placement
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=None,
    )

    # Prepare model for k-bit/LoRA training (critical for dtype consistency)
    model = prepare_model_for_kbit_training(model)


    # Apply LoRA
    print("\n" + "="*60)
    print("Applying LoRA adapters...")
    print("="*60)

    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        target_modules=model_config.lora_target_modules,
        task_type=model_config.lora_task_type,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✅ LoRA Applied Successfully!")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print("="*60 + "\n")

    if trainable_params == 0:
        raise RuntimeError("❌ CRITICAL ERROR: No trainable parameters after applying LoRA!")

    # Load dataset
    train_data_path = "path/to/your/training/data.json"  # Replace with your training data path
    with open(train_data_path, "r") as f:
        raw_data = json.load(f)
    
    dataset = GRPODataset(raw_data, tokenizer)
    
    output_dir = "path/to/save/trained/model"  # Replace with your desired output directory
    print("Output Dir:", output_dir)
    
    # Configure training with proper FSDP settings
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        warmup_steps=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_grad_norm=10.0,
        temperature=0.7,
        beta=0.04,
        scale_rewards="none",
        gradient_checkpointing=True,
        # bf16=False,
        # bf16_full_eval=False,
        save_steps=100,
        logging_steps=1,
        remove_unused_columns=False,  # ✅ CRITICAL: Keep all dataset columns for reward function
        # ✅ FSDP-specific settings for dtype consistency
        dataloader_num_workers=0,  # Avoid multiprocessing issues with FSDP
    )
    
    # Create reward function
    reward_function = create_reward_function(judge)
    
    trainer = GRPOTrainer(
        reward_funcs=reward_function,  # ✅ Correct parameter name
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting training with BF16 Autocast...")
    with autocast(dtype=torch.bfloat16):
        trainer.train()