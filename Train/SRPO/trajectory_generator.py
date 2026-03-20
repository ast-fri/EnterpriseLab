# trajectory_generator.py
class ReActTrajectoryGenerator:
    """Pure trajectory generation - NO distributed logic."""
    
    def __init__(self, model, tokenizer, tools, max_steps=10):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_steps = max_steps
    
    def generate(self, prompt: str, temperature=0.7) -> List[Dict]:
        """Generate single trajectory (synchronous, no FSDP logic)."""
        trajectory = []
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        
        for step in range(self.max_steps):
            # Generate thought+action
            generated = self._generate_completion(messages, temperature)
            thought, action = self._parse_output(generated)
            
            # Execute tool
            observation = self._execute_action(action)
            
            trajectory.append({
                "step": step,
                "thought": thought,
                "action": action,
                "observation": observation
            })
            
            # Update context
            messages.extend([
                {"role": "assistant", "content": generated},
                {"role": "user", "content": f"Observation: {observation}"}
            ])
            
            if self._is_complete(action, observation):
                break
        
        return trajectory
    
    def _generate_completion(self, messages: List[Dict], temperature: float) -> str:
        """Simple generation without distributed logic."""
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    def _parse_output(self, text: str) -> Tuple[str, str]:
        """Extract thought and action."""
        thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
        action_match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        
        return thought, action
    
    def _execute_action(self, action: str) -> str:
        """Execute tool call."""
        try:
            action_dict = json.loads(action)
            tool_name = action_dict.get("tool_name")
            tool_args = action_dict.get("tool_arguments", {})
            
            if hasattr(self.tools, tool_name):
                return getattr(self.tools, tool_name)(**tool_args)
            return f"Tool {tool_name} not found"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _is_complete(self, action: str, observation: str) -> bool:
        """Check completion."""
        completion_signals = ["final answer", "task complete", "done"]
        text = f"{action} {observation}".lower()
        return any(signal in text for signal in completion_signals)
