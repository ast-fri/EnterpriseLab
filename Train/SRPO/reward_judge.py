# reward_judge.py
class LLMRewardJudge:
    """Compute rewards using LLM - runs sequentially on main process only."""
    
    def __init__(self, judge_llm):
        self.judge_llm = judge_llm
    
    def score(self, trajectory: List[Dict], reference: List[Dict]) -> float:
        """Score single trajectory (no distributed logic)."""
        prompt = f"""
        Generated trajectory: {json.dumps(trajectory, indent=2)}
        Reference trajectory: {json.dumps(reference, indent=2)}
        
        Score this trajectory on a scale of 0-5...
        """
        
        response = self.judge_llm.gpt(prompt).content.strip()
        score_dict = self._parse_score(response)
        return float(score_dict.get("Score", 3.0))
    
    def _parse_score(self, response: str) -> Dict:
        """Parse LLM response."""
        if response.startswith("```"):
            response = response.split("```")[1].strip()
        try:
            return json.loads(response)
        except:
            return {"Score": "3"}
