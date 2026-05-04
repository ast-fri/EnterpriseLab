# scoring_engine.py
"""
Node interestingness scoring for AutoQuest Phase 2
Combines heuristic and LLM-based scoring strategies
"""

from typing import Dict, Optional, Any
import json
import random
from AutoQuest.intelligent_explorer.data_models import NodeExecution, EnvironmentKnowledgeBase
from AutoQuest.intelligent_explorer.tool_classifier import ToolOperation
from utils import normalize_operation


class ScoringEngine:
    """Scores node executions and exploration potential"""
    
    def __init__(
        self,
        tool_classifications: Dict,
        gpt_caller: callable,
        use_llm_scoring: bool = False,
        llm_sampling_rate: float = 0.2,
        persistence_manager: Optional[Any] = None
    ):
        """
        Args:
            tool_classifications: Tool classification mappings
            gpt_caller: Async function for GPT API calls
            use_llm_scoring: Whether to use LLM for scoring
            llm_sampling_rate: Probability of using LLM (to reduce costs)
            persistence_manager: PersistenceManager for serialization
        """
        self.tool_classifications = tool_classifications
        self.gpt_caller = gpt_caller
        self.use_llm_scoring = use_llm_scoring
        self.llm_sampling_rate = llm_sampling_rate
        self.persistence = persistence_manager
        
        # Track execution history for novelty scoring
        self.node_execution_history: Dict[str, list] = {}
        
        # Scoring weights
        self.weights = {
            "success": 0.3,
            "novelty": 0.3,
            "richness": 0.2,
            "resource_creation": 0.2
        }
    
    # ========================
    # Main Scoring Methods
    # ========================
    
    async def score_node_interestingness(
        self,
        execution: NodeExecution,
        node_name: str,
        kb: Optional[EnvironmentKnowledgeBase] = None
    ) -> float:
        """
        Score completed node execution for interestingness
        
        Args:
            execution: NodeExecution result
            node_name: Name of the node
            kb: Knowledge base context
            
        Returns:
            Score between 0.0 and 1.0
        """
        # Track execution
        if node_name not in self.node_execution_history:
            self.node_execution_history[node_name] = []
        self.node_execution_history[node_name].append(execution)
        
        # Compute heuristic score
        heuristic_score = self._compute_heuristic_score(execution, node_name, kb)
        
        # Optionally enhance with LLM scoring
        if self.use_llm_scoring and random.random() < self.llm_sampling_rate:
            classification = self.tool_classifications.get(node_name, {})
            llm_score = await self._llm_score_interestingness(
                execution, node_name, classification
            )
            
            # Blend heuristic and LLM scores
            final_score = heuristic_score * 0.7 + llm_score * 0.3
        else:
            final_score = heuristic_score
        
        return min(final_score, 1.0)
    
    async def score_next_node_potential(
        self,
        node_name: str,
        current_execution: NodeExecution,
        kb: EnvironmentKnowledgeBase
    ) -> float:
        """
        Score potential of exploring a node next
        
        Args:
            node_name: Candidate node name
            current_execution: Current execution context
            kb: Knowledge base state
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        classification = self.tool_classifications.get(node_name, {})
        operation = normalize_operation(classification.get("operation"))
        
        # 1. Operation type bonus (CREATE > LIST > UPDATE > READ > DELETE)
        operation_scores = {
            ToolOperation.CREATE: 0.4,
            ToolOperation.LIST: 0.3,
            ToolOperation.SEARCH: 0.25,
            ToolOperation.UPDATE: 0.25,
            ToolOperation.READ: 0.2,
            ToolOperation.DELETE: 0.15,
            ToolOperation.UNKNOWN: 0.1
        }
        score += operation_scores.get(operation, 0.1)
        
        # 2. Novelty bonus (less explored = higher score)
        exec_count = len(self.node_execution_history.get(node_name, []))
        novelty = max(0.0, 1.0 - exec_count / 20.0)  # Diminishes after 20 executions
        score += novelty * 0.3
        
        # 3. Resource richness bonus (more resources available = higher potential)
        if kb and kb.resource_by_type:
            resource_count = len(kb.resources)
            richness = min(resource_count / 10.0, 0.2)  # Cap at 0.2
            score += richness
        
        # 4. Dependency satisfaction bonus (has all required resources)
        if kb:
            required_resources = classification.get("requires_resources", [])
            if required_resources:
                satisfied_count = sum(
                    1 for res_type in required_resources
                    if res_type in kb.resource_by_type and kb.resource_by_type[res_type]
                )
                satisfaction_ratio = satisfied_count / len(required_resources)
                score += satisfaction_ratio * 0.1
        
        return min(score, 1.0)
    
    # ========================
    # Heuristic Scoring
    # ========================
    
    def _compute_heuristic_score(
        self,
        execution: NodeExecution,
        node_name: str,
        kb: Optional[EnvironmentKnowledgeBase]
    ) -> float:
        """
        Compute heuristic-based interestingness score
        
        Args:
            execution: NodeExecution result
            node_name: Name of the node
            kb: Knowledge base context
            
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # 1. Success bonus
        if execution.success:
            score += self.weights["success"]
        
        # 2. Novelty bonus (diminishes with repeated executions)
        exec_count = len(self.node_execution_history.get(node_name, []))
        novelty_score = max(0.0, 1.0 - exec_count / 10.0)
        score += novelty_score * self.weights["novelty"]
        
        # 3. Information richness
        if execution.tool_output:
            if isinstance(execution.tool_output, dict):
                # More fields = richer information
                field_count = len(execution.tool_output.keys())
                richness = min(field_count / 10.0, 1.0)
                score += richness * self.weights["richness"]
            elif isinstance(execution.tool_output, list):
                # More items = richer
                item_count = len(execution.tool_output)
                richness = min(item_count / 5.0, 1.0)
                score += richness * self.weights["richness"]
            else:
                # Some output is better than none
                score += 0.1 * self.weights["richness"]
        
        # 4. Resource creation bonus
        classification = self.tool_classifications.get(node_name, {})
        operation = normalize_operation(classification.get("operation"))
        
        if operation == ToolOperation.CREATE and execution.success:
            score += self.weights["resource_creation"]
        
        return score
    
    # ========================
    # LLM-Based Scoring
    # ========================
    
    async def _llm_score_interestingness(
        self,
        execution: NodeExecution,
        node_name: str,
        classification: Dict
    ) -> float:
        """
        Use LLM to score interestingness
        
        Args:
            execution: NodeExecution result
            node_name: Name of the node
            classification: Tool classification
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Prepare inputs/outputs for prompt
            inputs_str = json.dumps(execution.tool_input, default=str)[:300]
            
            if self.persistence:
                outputs_str = json.dumps(
                    self.persistence.serialize_output(execution.tool_output),
                    default=str
                )[:300]
            else:
                outputs_str = str(execution.tool_output)[:300]
            
            prompt = f"""Rate the interestingness of this tool execution on a scale from 0.0 to 1.0.

**Tool Information:**
- Name: {node_name}
- Operation: {classification.get('operation', 'unknown')}
- Success: {execution.success}

**Execution Details:**
- Inputs: {inputs_str}
- Outputs: {outputs_str}
- Execution Time: {execution.execution_time:.2f}s

**Rating Criteria:**
1. **Novelty**: Is this a unique or repetitive action?
2. **Information Richness**: Does the output contain valuable data?
3. **Downstream Potential**: Can this enable interesting follow-up actions?
4. **Practical Usefulness**: Is this a realistic, valuable operation?

**Instructions:**
- Consider all four criteria
- Return ONLY a single float between 0.0 and 1.0
- No explanation, just the number

Score:"""

            response = await self.gpt_caller(
                prompt=prompt,
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=10
            )
            
            # Parse score
            score_str = response.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"      ⚠️  LLM scoring error: {e}")
            return 0.5  # Default to neutral score on error
    
    # ========================
    # Diversity & Coverage Metrics
    # ========================
    
    def compute_diversity_score(self, kb: EnvironmentKnowledgeBase) -> float:
        """
        Compute diversity score based on KB state
        
        Args:
            kb: Knowledge base
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        if not kb.resource_by_type:
            return 0.0
        
        # Diversity = number of different resource types explored
        type_count = len(kb.resource_by_type)
        
        # Also consider distribution (avoid single-type dominance)
        total_resources = len(kb.resources)
        if total_resources == 0:
            return 0.0
        
        type_distribution = [
            len(kb.resource_by_type[res_type]) / total_resources
            for res_type in kb.resource_by_type
        ]
        
        # Shannon entropy as diversity metric
        import math
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in type_distribution)
        max_entropy = math.log2(type_count) if type_count > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine type count and distribution
        diversity = (type_count / 10.0) * 0.5 + normalized_entropy * 0.5
        
        return min(diversity, 1.0)
    
    def compute_coverage_score(
        self,
        executed_tools: set,
        total_tools: int
    ) -> float:
        """
        Compute exploration coverage score
        
        Args:
            executed_tools: Set of executed tool names
            total_tools: Total number of tools available
            
        Returns:
            Coverage score between 0.0 and 1.0
        """
        if total_tools == 0:
            return 0.0
        
        return len(executed_tools) / total_tools
    
    # ========================
    # Statistics
    # ========================
    
    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        total_executions = sum(len(execs) for execs in self.node_execution_history.values())
        
        return {
            "total_executions": total_executions,
            "unique_tools_executed": len(self.node_execution_history),
            "most_executed_tool": max(
                self.node_execution_history.items(),
                key=lambda x: len(x[1])
            )[0] if self.node_execution_history else None,
            "execution_distribution": {
                tool: len(execs)
                for tool, execs in sorted(
                    self.node_execution_history.items(),
                    key=lambda x: len(x[1]),
                    reverse=True
                )[:10]  # Top 10
            }
        }
    
    def reset_history(self):
        """Reset execution history"""
        self.node_execution_history.clear()
    
    def update_weights(self, weights: Dict[str, float]):
        """
        Update scoring weights
        
        Args:
            weights: Dictionary of weight updates
        """
        self.weights.update(weights)
        
        # Ensure weights are normalized
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
