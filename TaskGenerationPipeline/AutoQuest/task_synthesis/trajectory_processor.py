# # task_synthesis/trajectory_processor.py (REFACTORED v2 - WITH GROUND TRUTH)

# from typing import List, Dict, Any, Set, Tuple, Optional
# from dataclasses import dataclass, field
# from collections import defaultdict
# import json
# from enum import Enum


# # ============================================================================
# # DATA STRUCTURES
# # ============================================================================

# @dataclass
# class Step:
#     """Represents a single step in a trajectory."""
#     step_number: int
#     depth: int
#     tool_name: str
#     parent_node: Optional[str]
#     inputs: Dict[str, Any]
#     outputs: Any
#     success: bool
#     error: bool
#     execution_time: float
#     interestingness_score: float
#     kb_version: int

#     def get_input_entities(self) -> Set[str]:
#         """Extract entity IDs from inputs."""
#         entities = set()

#         if not self.inputs:
#             return entities

#         def extract_from_value(value):
#             if isinstance(value, str):
#                 entities.add(value)
#             elif isinstance(value, list):
#                 for item in value:
#                     extract_from_value(item)
#             elif isinstance(value, dict):
#                 for v in value.values():
#                     extract_from_value(v)
#             elif isinstance(value, (int, float, bool)):
#                 entities.add(str(value))
#             elif isinstance(value, tuple):
#                 for item in value:
#                     extract_from_value(item)
#             elif value is not None:
#                 try:
#                     entities.add(str(value))
#                 except Exception:
#                     pass

#         for key, value in self.inputs.items():
#             extract_from_value(value)

#         return entities

#     def get_output_entities(self) -> Set[str]:
#         """Extract entity IDs from outputs."""
#         entities = set()

#         if not self.outputs:
#             return entities

#         def extract_from_value(value):
#             if isinstance(value, str):
#                 entities.add(value)
#             elif isinstance(value, list):
#                 for item in value:
#                     extract_from_value(item)
#             elif isinstance(value, dict):
#                 for v in value.values():
#                     extract_from_value(v)
#             elif isinstance(value, (int, float, bool)):
#                 entities.add(str(value))
#             elif isinstance(value, tuple):
#                 for item in value:
#                     extract_from_value(item)
#             elif value is not None:
#                 try:
#                     entities.add(str(value))
#                 except Exception:
#                     pass

#         if isinstance(self.outputs, dict):
#             for key, value in self.outputs.items():
#                 extract_from_value(value)
#         elif isinstance(self.outputs, list):
#             for item in self.outputs:
#                 extract_from_value(item)
#         else:
#             extract_from_value(self.outputs)

#         return entities


# @dataclass
# class FilteredSequence:
#     """Represents a filtered meaningful sequence ready for task generation."""
#     sequence_id: str
#     steps: List[Step]  # Only interesting steps
#     prerequisite_context: List[Step]  # Steps before first interesting step (for context)
#     domain: str
#     data_flow: Dict[int, List[int]]  # step_number -> dependent step_numbers
#     entry_point: str  # First tool
#     exit_point: str  # Last tool


# @dataclass
# class AnnotatedStep:
#     """A step with generated rationale (low-level task) and ground truth output."""
#     step_number: int
#     tool_name: str
#     rationale: str  # The "thought" or low-level task description
#     inputs: Dict[str, Any]
#     outputs: Any  # UPDATED: This is the ground truth output for this low-level task
#     expected_output: Any  # NEW: Alias for clarity (same as outputs)
#     entities_used: Set[str]


# @dataclass
# class SyntheticTask:
#     """The final output: a high-level task with reasoning chain and ground truth."""
#     task_id: str
#     instruction: str  # High-level user instruction
#     chain_of_thought: List[AnnotatedStep]  # Low-level reasoning steps with their outputs
#     required_tools: List[str]
#     success_criteria: List[str]
#     domain: str
#     difficulty: str
#     prerequisite_context: List[Dict[str, Any]] = field(default_factory=list)
#     ground_truth: Any = None  # NEW: Expected final output of the entire task
#     meta: Dict[str, Any] = field(default_factory=dict)


# # ============================================================================
# # COMBINED STEP 1 + 2: INTELLIGENT SEQUENCE EXTRACTION
# # ============================================================================

# class IntelligentSequenceExtractor:
#     """
#     Combines clustering and subsequence extraction with interestingness filtering.
#     Extracts all meaningful sequences from a trajectory with LLM-based filtering.
#     """

#     def __init__(self, llm_caller, interestingness_threshold: float = 0.5):
#         self.llm = llm_caller
#         self.threshold = interestingness_threshold
#         self.domain_keywords = {
#             'CRM': ['customer', 'lead', 'opportunity', 'contact', 'account'],
#             'HR': ['employee', 'emp_id', 'DOJ', 'DOL', 'salary', 'leave'],
#             'IT': ['ticket', 'issue', 'resolution', 'assigned'],
#         }

#     async def extract_sequences(self, trajectory: Dict) -> List[FilteredSequence]:
#         """
#         Main method: Extract all meaningful sequences from trajectory.
#         Returns filtered sequences ready for task generation.
#         """
#         # Parse trajectory steps
#         steps = [Step(**step_data) for step_data in trajectory['steps']]

#         if not steps:
#             return []

#         print(f"[SequenceExtractor] Processing {len(steps)} steps")

#         # Generate all possible sequences (Option C)
#         all_sequences = []

#         # For each starting position
#         for start_idx in range(len(steps)):
#             # For each ending position from start
#             for end_idx in range(start_idx + 1, len(steps) + 1):
#                 sequence_steps = steps[start_idx:end_idx]

#                 # Collect prerequisite context (steps before start_idx)
#                 prerequisite_steps = steps[:start_idx] if start_idx > 0 else []

#                 all_sequences.append({
#                     'steps': sequence_steps,
#                     'prerequisites': prerequisite_steps,
#                     'start_idx': start_idx,
#                     'end_idx': end_idx
#                 })

#         print(f"[SequenceExtractor] Generated {len(all_sequences)} candidate sequences")

#         # Score and filter each sequence
#         filtered_sequences = []

#         for idx, seq_data in enumerate(all_sequences):
#             print(f"[SequenceExtractor] Processing sequence {idx+1}/{len(all_sequences)}")

#             # Score interestingness for each step in this sequence context
#             scored_steps = await self._score_sequence_interestingness(
#                 seq_data['steps'],
#                 seq_data['prerequisites']
#             )

#             # Filter out uninteresting steps with data flow check
#             filtered_steps = self._filter_by_interestingness_and_dataflow(scored_steps)

#             # Only keep sequences with at least 1 interesting step
#             if filtered_steps:
#                 # Check if first filtered step is above threshold (valid starting point)
#                 if filtered_steps[0].interestingness_score >= self.threshold:
#                     domain = self._infer_domain(filtered_steps)
#                     data_flow = self._build_data_flow_graph(filtered_steps)

#                     sequence_id = f"seq_{seq_data['start_idx']}_{seq_data['end_idx']}_filtered_{len(filtered_steps)}"

#                     filtered_seq = FilteredSequence(
#                         sequence_id=sequence_id,
#                         steps=filtered_steps,
#                         prerequisite_context=seq_data['prerequisites'],
#                         domain=domain,
#                         data_flow=data_flow,
#                         entry_point=filtered_steps[0].tool_name,
#                         exit_point=filtered_steps[-1].tool_name
#                     )

#                     filtered_sequences.append(filtered_seq)
#                     print(f"  ✓ Kept sequence with {len(filtered_steps)} interesting steps")
#                 else:
#                     print(f"  ✗ Skipped: first step not interesting enough")
#             else:
#                 print(f"  ✗ Skipped: no interesting steps")

#         print(f"[SequenceExtractor] Extracted {len(filtered_sequences)} meaningful sequences")
#         return filtered_sequences

#     async def _score_sequence_interestingness(
#         self,
#         steps: List[Step],
#         prerequisite_steps: List[Step]
#     ) -> List[Step]:
#         """
#         Score each step's interestingness based on context.
#         Context includes: previous steps in sequence + prerequisite steps.
#         """
#         scored_steps = []

#         for idx, step in enumerate(steps):
#             # Build context: prerequisite steps + previous steps in sequence
#             context_steps = prerequisite_steps + steps[:idx]

#             # Call LLM to score interestingness
#             score = await self._call_interestingness_llm(step, context_steps)

#             # Create new Step object with updated score
#             scored_step = Step(
#                 step_number=step.step_number,
#                 depth=step.depth,
#                 tool_name=step.tool_name,
#                 parent_node=step.parent_node,
#                 inputs=step.inputs,
#                 outputs=step.outputs,
#                 success=step.success,
#                 error=step.error,
#                 execution_time=step.execution_time,
#                 interestingness_score=score,
#                 kb_version=step.kb_version
#             )

#             scored_steps.append(scored_step)
#             print(f"    Step {step.step_number} ({step.tool_name}): score = {score:.2f}")

#         return scored_steps

#     async def _call_interestingness_llm(
#         self,
#         current_step: Step,
#         context_steps: List[Step]
#     ) -> float:
#         """
#         Call LLM to determine interestingness score for a step.
#         Returns: float between 0.0 and 1.0
#         """
#         prompt = self._build_interestingness_prompt(current_step, context_steps)

#         try:
#             response = await self.llm(prompt, response_format="json")
#             score = float(response.get("interestingness_score", 0.5))

#             # Penalize if there's an error
#             if current_step.error:
#                 score = max(0.0, score - 0.3)

#             # Clamp to [0, 1]
#             return max(0.0, min(1.0, score))

#         except Exception as e:
#             print(f"    Warning: LLM call failed ({str(e)}), using default score 0.5")
#             return 0.5 if not current_step.error else 0.2

#     def _build_interestingness_prompt(
#         self,
#         current_step: Step,
#         context_steps: List[Step]
#     ) -> str:
#         """Build prompt for interestingness scoring."""

#         # Format context
#         context_text = ""
#         if context_steps:
#             context_text = "\n".join([
#                 f"- Step {s.step_number}: {s.tool_name} | Inputs: {json.dumps(s.inputs)[:100]} | Outputs: {json.dumps(s.outputs)[:100]}"
#                 for s in context_steps[-3:]  # Last 3 for brevity
#             ])
#         else:
#             context_text = "This is the first step (no prior context)"

#         error_note = ""
#         if current_step.error:
#             error_note = "\n⚠️ WARNING: This step resulted in an ERROR. This should significantly reduce interestingness."

#         prompt = f"""You are evaluating whether a step in an agentic workflow is "interesting" and valuable for training data generation.

# CONTEXT (Previous steps in this workflow):
# {context_text}

# CURRENT STEP TO EVALUATE:
# Tool: {current_step.tool_name}
# Inputs: {json.dumps(current_step.inputs, indent=2)}
# Outputs: {json.dumps(current_step.outputs, indent=2)}
# Success: {current_step.success}
# {error_note}

# TASK: Assign an "interestingness score" from 0.0 to 1.0 based on:

# HIGH INTERESTINGNESS (0.7-1.0):
# - Introduces new, meaningful entities or data
# - Performs critical business logic (create, update, delete important records)
# - Has clear dependencies on previous steps
# - Outputs are used by subsequent steps
# - Represents a key decision point

# MEDIUM INTERESTINGNESS (0.4-0.6):
# - Retrieves or validates existing data
# - Performs routine operations
# - Supports main workflow but not critical
# - Some connection to context

# LOW INTERESTINGNESS (0.0-0.3):
# - Redundant or trivial operations
# - No clear purpose given context
# - Errors or failures (penalize heavily)
# - Outputs not useful

# OUTPUT FORMAT (JSON):
# {{
#   "interestingness_score": 0.75,
#   "reasoning": "Brief explanation (1 sentence)"
# }}

# YOUR EVALUATION:"""

#         return prompt

#     def _filter_by_interestingness_and_dataflow(
#         self,
#         steps: List[Step]
#     ) -> List[Step]:
#         """
#         Filter steps by interestingness threshold with data flow checks.
#         Skips uninteresting steps but includes next interesting step if data flow exists.
#         """
#         if not steps:
#             return []

#         filtered = []
#         last_interesting_step = None

#         for step in steps:
#             if step.interestingness_score >= self.threshold:
#                 # This step is interesting

#                 # If we have a previous interesting step, check data flow
#                 if last_interesting_step is not None:
#                     # Check if there's data flow from last interesting to current
#                     has_data_flow = self._check_data_flow(last_interesting_step, step)

#                     if has_data_flow:
#                         filtered.append(step)
#                         last_interesting_step = step
#                     else:
#                         # No data flow, this breaks the chain
#                         # Don't include this step
#                         print(f"      Skipping step {step.step_number}: no data flow from break point")
#                 else:
#                     # First interesting step
#                     filtered.append(step)
#                     last_interesting_step = step
#             else:
#                 # Uninteresting step - skip it
#                 # last_interesting_step remains as the "break point"
#                 print(f"      Skipping step {step.step_number}: score {step.interestingness_score:.2f} < {self.threshold}")

#         return filtered

#     def _check_data_flow(self, step_a: Step, step_b: Step) -> bool:
#         """
#         Check if there's data flow from step_a to step_b.
#         Returns True if step_b's inputs contain entities from step_a's outputs.
#         """
#         output_entities = step_a.get_output_entities()
#         input_entities = step_b.get_input_entities()

#         # Check for intersection
#         common = output_entities & input_entities

#         return len(common) > 0

#     def _build_data_flow_graph(self, steps: List[Step]) -> Dict[int, List[int]]:
#         """Build data flow graph: step_number -> list of dependent step_numbers."""
#         flow = defaultdict(list)

#         for i, step_a in enumerate(steps):
#             output_entities = step_a.get_output_entities()

#             for j, step_b in enumerate(steps[i+1:], start=i+1):
#                 input_entities = step_b.get_input_entities()

#                 if output_entities & input_entities:
#                     flow[step_a.step_number].append(step_b.step_number)

#         return flow

#     def _infer_domain(self, steps: List[Step]) -> str:
#         """Infer domain based on tool names and data."""
#         domain_scores = defaultdict(int)

#         for step in steps:
#             step_text = json.dumps(step.inputs) + step.tool_name
#             for domain, keywords in self.domain_keywords.items():
#                 for keyword in keywords:
#                     if keyword.lower() in step_text.lower():
#                         domain_scores[domain] += 1

#         return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "General"


# # ============================================================================
# # STEP 3: LOW-LEVEL TASK GENERATION (with ground truth outputs)
# # ============================================================================

# class LowLevelTaskGenerator:
#     """Generates rationales for each step with prerequisite context and ground truth outputs."""

#     def __init__(self, llm_caller):
#         self.llm = llm_caller

#     async def generate_rationales(
#         self,
#         filtered_sequence: FilteredSequence
#     ) -> List[AnnotatedStep]:
#         """
#         Generate rationale for each step, using prerequisite context.
#         Each AnnotatedStep includes the actual output as ground truth.
#         """
#         annotated_steps = []

#         # Build prerequisite context text
#         prereq_context = self._format_prerequisite_context(
#             filtered_sequence.prerequisite_context
#         )

#         for idx, step in enumerate(filtered_sequence.steps):
#             prompt = self._build_rationale_prompt(
#                 step,
#                 annotated_steps,  # Previous annotated steps
#                 prereq_context,
#                 filtered_sequence.data_flow.get(step.step_number, []),
#                 filtered_sequence.steps
#             )

#             try:
#                 response = await self.llm(prompt, response_format="text")
#                 print("Low level task rationale: ",response)
#                 rationale = response.strip() if isinstance(response, str) else response.get("response", "")
#                  # Fallback if still empty
#                 if not rationale:
#                     rationale = f"I will execute {step.tool_name} to process the required data."
#             except Exception as e:
#                 rationale = f"Execute {step.tool_name} to progress the workflow."

#             # NEW: Include the actual output as ground truth
#             annotated = AnnotatedStep(
#                 step_number=step.step_number,
#                 tool_name=step.tool_name,
#                 rationale=rationale,
#                 inputs=step.inputs,
#                 outputs=step.outputs,  # Ground truth from trajectory
#                 expected_output=step.outputs,  # Explicit alias for clarity
#                 entities_used=step.get_input_entities() | step.get_output_entities()
#             )
#             annotated_steps.append(annotated)

#         return annotated_steps

#     def _format_prerequisite_context(self, prereq_steps: List[Step]) -> str:
#         """Format prerequisite steps as context string."""
#         if not prereq_steps:
#             return "No prior context - this task starts from the beginning."

#         context_lines = ["PREREQUISITE CONTEXT (completed before this task):"]
#         for step in prereq_steps:
#             context_lines.append(
#                 f"- {step.tool_name}: {json.dumps(step.inputs)[:80]} → {json.dumps(step.outputs)[:80]}"
#             )

#         return "\n".join(context_lines)

#     def _build_rationale_prompt(
#         self,
#         step: Step,
#         previous_annotated: List[AnnotatedStep],
#         prereq_context: str,
#         dependencies: List[int],
#         all_steps: List[Step]
#     ) -> str:
#         """Refined prompt to ensure non-empty, action-oriented rationales."""
        
#         # improved context formatting
#         prev_action = "None (Start of task)"
#         if previous_annotated:
#             last = previous_annotated[-1]
#             prev_action = f"Executed {last.tool_name} (Step {last.step_number})"

#         dep_details = []
#         for dep_step_num in dependencies:
#             dep_step = next((s for s in all_steps if s.step_number == dep_step_num), None)
#             if dep_step:
#                 dep_details.append(f"- Output from {dep_step.tool_name} (Step {dep_step_num})")
        
#         dependency_text = "\n".join(dep_details) if dep_details else "No direct data dependency from previous steps in this sequence."

#         prompt = f"""You are an AI agent executing a workflow. You must explain your NEXT step.

# CONTEXT:
# Prerequisites completed: 
# {prereq_context}

# Last Action: {prev_action}

# CURRENT GOAL:
# Execute Tool: {step.tool_name}
# With Inputs: {json.dumps(step.inputs, indent=2)}

# DEPENDENCIES:
# {dependency_text}

# INSTRUCTIONS:
# 1. Write a single first-person sentence explaining WHAT you are doing and WHICH data you are using and WHY are you doing it.
# - Explicitly mention key values from the inputs (IDs, names, dates).
# - Explain the immediate purpose (e.g., "creating a record for...", "updating the status of...").
# - Do NOT hallucinate intentions not visible in the data.
# 2. Add the details of ALL the input arguments and their values into the step rationale.

# BAD: "I will analyze the data." (Vague)
# BAD: "I will call the tool." (Robotic)
# GOOD: "I am creating a new product record for 'Smartphone X' with ID product_001 and setting its price to $599."

# YOUR RATIONALE:"""
        
#         return prompt


# # ============================================================================
# # STEP 4: HIGH-LEVEL TASK GENERATION (with ground truth output)
# # ============================================================================

# class HighLevelTaskGenerator:
#     """Synthesizes high-level instruction from chain of thought with ground truth."""

#     def __init__(self, llm_caller):
#         self.llm = llm_caller

#     async def generate_instruction(
#         self,
#         annotated_steps: List[AnnotatedStep],
#         prerequisite_context: List[Step],
#         domain: str,
#         sequence_id: str
#     ) -> SyntheticTask:
#         """
#         Generate high-level task instruction with ground truth output.
#         Ground truth = final step's output or aggregated result.
#         """
#         prompt = self._build_instruction_prompt(annotated_steps, prerequisite_context, domain)

#         try:
#             response = await self.llm(prompt, response_format="json")
#             instruction = response.get("instruction", "")
#             success_criteria = response.get("success_criteria", [])
#         except Exception as e:
#             instruction = f"Complete a {domain} workflow involving {len(annotated_steps)} steps."
#             success_criteria = [f"All {len(annotated_steps)} actions complete successfully"]

#         required_tools = list(set(step.tool_name for step in annotated_steps))
#         difficulty = self._assess_difficulty(annotated_steps)

#         task_id = f"task_{sequence_id}"

#         # Format prerequisite context for task
#         prereq_data = [
#             {
#                 "step_number": s.step_number,
#                 "tool": s.tool_name,
#                 "inputs": s.inputs,
#                 "outputs": s.outputs
#             }
#             for s in prerequisite_context
#         ]

#         # NEW: Determine ground truth for the entire task
#         ground_truth = self._determine_task_ground_truth(annotated_steps)

#         return SyntheticTask(
#             task_id=task_id,
#             instruction=instruction,
#             chain_of_thought=annotated_steps,
#             required_tools=required_tools,
#             success_criteria=success_criteria,
#             domain=domain,
#             difficulty=difficulty,
#             prerequisite_context=prereq_data,
#             ground_truth=ground_truth,  # NEW: Task-level ground truth
#             meta={
#                 "num_steps": len(annotated_steps),
#                 "num_prerequisites": len(prerequisite_context),
#                 "entry_tool": annotated_steps[0].tool_name,
#                 "exit_tool": annotated_steps[-1].tool_name
#             }
#         )

#     def _determine_task_ground_truth(self, annotated_steps: List[AnnotatedStep]) -> Dict[str, Any]:
#         """
#         Determine the ground truth output for the entire task.

#         Strategy:
#         1. If single step: return that step's output
#         2. If multiple steps: return final step's output + aggregated outputs from all steps
#         """
#         if not annotated_steps:
#             return {}

#         # Final step output (most important)
#         final_output = annotated_steps[-1].outputs

#         # Aggregate all step outputs for comprehensive ground truth
#         all_step_outputs = []
#         for step in annotated_steps:
#             all_step_outputs.append({
#                 "step_number": step.step_number,
#                 "tool": step.tool_name,
#                 "output": step.outputs
#             })

#         ground_truth = {
#             "final_output": final_output,  # The ultimate result
#             "all_step_outputs": all_step_outputs,  # Complete execution trace
#             "final_entities": list(annotated_steps[-1].entities_used)  # Entities in final state
#         }

#         return ground_truth

#     def _build_instruction_prompt(
#         self,
#         annotated_steps: List[AnnotatedStep],
#         prerequisite_context: List[Step],
#         domain: str
#     ) -> str:
#         """Build prompt for high-level instruction generation."""

#         cot_text = "\n".join([
#             f"{i+1}. {step.rationale}"
#             for i, step in enumerate(annotated_steps)
#         ])

#         prereq_text = ""
#         if prerequisite_context:
#             prereq_text = "\n\nPREREQUISITE STEPS (already completed):\n" + "\n".join([
#                 f"- {s.tool_name}: {json.dumps(s.inputs)[:60]}"
#                 for s in prerequisite_context[:3]
#             ])

#         all_entities = set()
#         for step in annotated_steps:
#             all_entities.update(step.entities_used)
#         entities_text = ", ".join(list(all_entities)[:5])

#         prompt = f"""You are creating a dataset of human instructions for an AI agent. You are reverse-engineering a user's original request from an AI agent's execution trace.

# DOMAIN: {domain}
# {prereq_text}

# EXECUTION TRACE (What the agent actually did):
# {cot_text}

# KEY ENTITIES: {entities_text}

# RULES FOR INSTRUCTION:
# 1. **Descriptive & Story-like**: Write it as a request from a manager or user. 
#    - BAD: "Create product product_001."
#    - GOOD: "We have a new item launching today. Please add 'Smartphone X' (ID: product_001) to our catalog under the electronics category with a price of $599."
# 2. **Data Completeness**: Every piece of data needed for the creation steps (names, IDs, prices, dates) MUST be embedded in the text naturally. If the agent needs 'customer_id_001' to run, you must mention it in the instruction.
# 3. **Tool input details in Task**: Include details of all the inputs of the tool, and that too perfect match with the argument keys, don't fabricate of change the key of the inputs arguments, also consider all the details in tool input to be synthesized in the task. It should be like that the the agent will be able to get all the details of tool input from the task itself. 
# 4. **No Technical Jargon**: Do not say "use the create_product tool". Say "register a new product" or "add a product". 
# 5. **Grounded**: Do NOT ask for analysis ("tell me how it's performing") if the tools only created data. If the trace shows creation, ask for creation.
# 6. **Parameter data/input argument data**: Based on the all the tools/steps in execution trace, include whatever data necessary to execute the task, like input arguments, if the arguments are part of output of some tool then only no need to include, else include the argument inputs in the text of the instruction.

# OUTPUT FORMAT (JSON):
# {{
#   "instruction": "The full paragraph instruction including all data points.",
#   "success_criteria": [
#     "Verifiable criterion 1",
#     "Verifiable criterion 2",
#     "Verifiable criterion 3"
#   ]
# }}

# YOUR RESPONSE:"""

#         return prompt

#     def _assess_difficulty(self, annotated_steps: List[AnnotatedStep]) -> str:
#         """Assess difficulty based on number of steps."""
#         num_steps = len(annotated_steps)

#         if num_steps <= 3:
#             return "EASY"
#         elif num_steps <= 6:
#             return "MEDIUM"
#         else:
#             return "HARD"


# # ============================================================================
# # ORCHESTRATION PIPELINE (Updated)
# # ============================================================================

# class CoTTaskSynthesisPipeline:
#     """Orchestrates the refactored pipeline with ground truth outputs."""

#     def __init__(self, llm_caller):
#         self.sequence_extractor = IntelligentSequenceExtractor(llm_caller, interestingness_threshold=0.5)
#         self.low_level_gen = LowLevelTaskGenerator(llm_caller)
#         self.high_level_gen = HighLevelTaskGenerator(llm_caller)

#     async def process_trajectory(
#         self,
#         trajectory: Dict,
#         tasks_output_dir: str,
#         max_tasks_per_cluster: int = 3
#     ) -> List[SyntheticTask]:
#         """Process trajectory through refactored pipeline."""
#         print(f"[Pipeline] Processing trajectory: {trajectory['trajectory_id']}")

#         # Combined Step 1+2: Extract intelligent sequences
#         print("[Pipeline] Extracting intelligent sequences (with interestingness filtering)...")
#         filtered_sequences = await self.sequence_extractor.extract_sequences(trajectory)
#         print(f"[Pipeline] Extracted {len(filtered_sequences)} meaningful sequences")

#         # Limit sequences to process
#         filtered_sequences = filtered_sequences[:max_tasks_per_cluster * 3]

#         all_tasks = []

#         for seq in filtered_sequences:
#             print(f"\n[Pipeline] Processing sequence: {seq.sequence_id}")
#             print(f"  Steps: {[s.step_number for s in seq.steps]}")
#             print(f"  Prerequisites: {[s.step_number for s in seq.prerequisite_context]}")

#             # Step 3: Generate low-level rationales (with ground truth outputs)
#             print("[Pipeline] Generating low-level rationales with ground truth...")
#             annotated_steps = await self.low_level_gen.generate_rationales(seq)
#             print(f"[Pipeline] Generated {len(annotated_steps)} rationales with outputs")

#             # Step 4: Generate high-level instruction (with task ground truth)
#             print("[Pipeline] Synthesizing high-level instruction with ground truth...")
#             task = await self.high_level_gen.generate_instruction(
#                 annotated_steps,
#                 seq.prerequisite_context,
#                 seq.domain,
#                 seq.sequence_id
#             )
#             print(f"[Pipeline] Generated task: {task.task_id}")
#             self.export_tasks(all_tasks, tasks_output_dir)
#             all_tasks.append(task)

#         print(f"\n[Pipeline] Complete! Generated {len(all_tasks)} tasks total")
#         return all_tasks

#     def export_tasks(self, tasks: List[SyntheticTask], output_path: str):
#         """Export tasks to JSON file with ground truth."""
#         output = []
#         for task in tasks:
#             output.append({
#                 "task_id": task.task_id,
#                 "instruction": task.instruction,
#                 "prerequisite_context": task.prerequisite_context,
#                 "chain_of_thought": [
#                     {
#                         "step": step.step_number,
#                         "rationale": step.rationale,
#                         "tool": step.tool_name,
#                         "inputs": step.inputs,
#                         "expected_output": step.expected_output  # NEW: Ground truth per step
#                     }
#                     for step in task.chain_of_thought
#                 ],
#                 "required_tools": task.required_tools,
#                 "success_criteria": task.success_criteria,
#                 "domain": task.domain,
#                 "difficulty": task.difficulty,
#                 "ground_truth": task.ground_truth,  # NEW: Task-level ground truth
#                 "meta": task.meta
#             })

#         with open(output_path, 'w') as f:
#             json.dump(output, f, indent=2)

#         print(f"[Pipeline] Exported {len(tasks)} tasks to {output_path}")



# task_synthesis/trajectory_processor_final.py
# Complete implementation with all quality control fixes

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import hashlib
import re
from itertools import combinations


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Step:
    """Represents a single step in a trajectory."""
    step_number: int
    depth: int
    tool_name: str
    parent_node: Optional[str]
    inputs: Dict[str, Any]
    outputs: Any
    success: bool
    error: bool
    execution_time: float
    interestingness_score: float
    kb_version: int

    def get_input_entities(self) -> Set[str]:
        entities = set()
        if not self.inputs: return entities

        def extract_from_value(value):
            if isinstance(value, str): entities.add(value)
            elif isinstance(value, list):
                for item in value: extract_from_value(item)
            elif isinstance(value, dict):
                for v in value.values(): extract_from_value(v)
            elif isinstance(value, (int, float, bool)): entities.add(str(value))
            elif isinstance(value, tuple):
                for item in value: extract_from_value(item)

        for key, value in self.inputs.items():
            extract_from_value(value)
        return entities

    def get_output_entities(self) -> Set[str]:
        entities = set()
        if not self.outputs: return entities

        def extract_from_value(value):
            if isinstance(value, str): entities.add(value)
            elif isinstance(value, list):
                for item in value: extract_from_value(item)
            elif isinstance(value, dict):
                for v in value.values(): extract_from_value(v)
            elif isinstance(value, (int, float, bool)): entities.add(str(value))
            elif isinstance(value, tuple):
                for item in value: extract_from_value(item)

        if isinstance(self.outputs, dict):
            for key, value in self.outputs.items(): extract_from_value(value)
        elif isinstance(self.outputs, list):
            for item in self.outputs: extract_from_value(item)
        else:
            extract_from_value(self.outputs)
        return entities


@dataclass
class FilteredSequence:
    sequence_id: str
    steps: List[Step]
    prerequisite_context: List[Step]
    domain: str
    data_flow: Dict[int, List[int]]
    entry_point: str
    exit_point: str


@dataclass
class AnnotatedStep:
    step_number: int
    tool_name: str
    rationale: str
    inputs: Dict[str, Any]
    outputs: Any
    expected_output: Any
    entities_used: Set[str]


@dataclass
class SyntheticTask:
    task_id: str
    instruction: str
    chain_of_thought: List[AnnotatedStep]
    required_tools: List[str]
    success_criteria: List[str]
    domain: str
    difficulty: str
    prerequisite_context: List[Dict[str, Any]] = field(default_factory=list)
    ground_truth: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# UTILITY: OUTPUT SANITIZER
# ============================================================================

class OutputSanitizer:
    """Sanitizes large/noisy outputs before inserting into prompts."""

    @staticmethod
    def sanitize(output: Any, max_chars: int = 200) -> str:
        """Convert output to clean, concise string."""
        if output is None:
            return "None"

        # Handle error outputs
        if isinstance(output, dict):
            if "error" in output or "error_human" in output:
                error_msg = output.get("error", output.get("error_human", "Unknown error"))
                return f"Error: {str(error_msg)[:max_chars]}"

            # Check for HTML error pages
            if "raw_response" in output:
                raw = str(output["raw_response"])
                if "<html" in raw.lower() or "<!doctype" in raw.lower():
                    # Extract meaningful error from HTML
                    error_match = re.search(r"<li class='error'>(.+?)</li>", raw, re.DOTALL)
                    if error_match:
                        error_text = re.sub(r'<[^>]+>', '', error_match.group(1))
                        return f"Error: {error_text.strip()[:max_chars]}"
                    return "Error: HTML error page returned"

        # Handle string outputs
        if isinstance(output, str):
            # Check if it's HTML
            if "<html" in output.lower() or "<!doctype" in output.lower():
                return "Error: HTML page returned (configuration issue)"

            # Check for "Error:" prefix
            if output.startswith("Error:"):
                return output[:max_chars]

        # Normal output - stringify and truncate
        output_str = json.dumps(output, default=str)
        if len(output_str) > max_chars:
            return output_str[:max_chars] + "..."
        return output_str

    @staticmethod
    def has_error(output: Any) -> bool:
        """Check if output represents an error."""
        if isinstance(output, dict):
            if "error" in output or "error_human" in output:
                return True
            if "raw_response" in output:
                raw = str(output.get("raw_response", ""))
                if "error" in raw.lower() or "<li class='error'>" in raw:
                    return True

        if isinstance(output, str):
            if output.startswith("Error:") or "error" in output.lower():
                return True

        return False


# ============================================================================
# STEP 1+2: INTELLIGENT SEQUENCE EXTRACTION (WITH FIXES)
# ============================================================================

class IntelligentSequenceExtractor:
    """
    Extracts meaningful sequences with:
    - Prerequisite filtering (only successful steps)
    - Output sanitization
    - Interestingness scoring
    """

    def __init__(self, llm_caller, interestingness_threshold: float = 0.5):
        self.llm = llm_caller
        self.threshold = interestingness_threshold
        self.sanitizer = OutputSanitizer()
        self.domain_keywords = {
            'CRM': ['customer', 'lead', 'opportunity', 'contact', 'account'],
            'HR': ['employee', 'emp_id', 'DOJ', 'DOL', 'salary', 'leave'],
            'IT': ['ticket', 'issue', 'resolution', 'assigned', 'support'],
            'Storage': ['file', 'folder', 'upload', 'download', 'cloud', 'owncloud'],
        }

    async def extract_sequences(self, trajectory: Dict) -> List[FilteredSequence]:
        """Main method: Extract meaningful sequences with quality control."""
        steps = [Step(**step_data) for step_data in trajectory['steps']]
        if not steps: return []

        print(f"[SequenceExtractor] Processing {len(steps)} steps")
        all_sequences = []

        # Generate all possible sequences (Option C)
        for start_idx in range(len(steps)):
            for end_idx in range(start_idx + 1, len(steps) + 1):
                sequence_steps = steps[start_idx:end_idx]

                # FIX 1: Filter prerequisites - only keep successful ones
                prerequisite_steps = [
                    s for s in steps[:start_idx] 
                    if s.success and not s.error
                ]

                all_sequences.append({
                    'steps': sequence_steps,
                    'prerequisites': prerequisite_steps,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })

        print(f"[SequenceExtractor] Generated {len(all_sequences)} candidate sequences")

        # Score and filter each sequence
        filtered_sequences = []

        for idx, seq_data in enumerate(all_sequences):
            # Score interestingness
            scored_steps = await self._score_sequence_interestingness(
                seq_data['steps'], seq_data['prerequisites']
            )

            # Filter by interestingness + data flow
            filtered_steps = self._filter_by_interestingness_and_dataflow(scored_steps)

            # Only keep if first step is interesting
            if filtered_steps and filtered_steps[0].interestingness_score >= self.threshold:
                domain = self._infer_domain(filtered_steps)
                data_flow = self._build_data_flow_graph(filtered_steps)
                sequence_id = f"seq_{seq_data['start_idx']}_{seq_data['end_idx']}_filtered_{len(filtered_steps)}"

                filtered_seq = FilteredSequence(
                    sequence_id=sequence_id,
                    steps=filtered_steps,
                    prerequisite_context=seq_data['prerequisites'],
                    domain=domain,
                    data_flow=data_flow,
                    entry_point=filtered_steps[0].tool_name,
                    exit_point=filtered_steps[-1].tool_name
                )

                filtered_sequences.append(filtered_seq)

        print(f"[SequenceExtractor] Extracted {len(filtered_sequences)} meaningful sequences")
        return filtered_sequences

    async def _score_sequence_interestingness(self, steps, prerequisites):
        """Score each step's interestingness based on context."""
        scored_steps = []
        for idx, step in enumerate(steps):
            context = prerequisites + steps[:idx]
            score = await self._call_interestingness_llm(step, context)

            # Update step with score
            step.interestingness_score = score
            scored_steps.append(step)
            print(f"    Step {step.step_number} ({step.tool_name}): score = {score:.2f}")
        return scored_steps

    async def _call_interestingness_llm(self, current_step, context_steps):
        """Call LLM to determine interestingness score."""
        prompt = self._build_interestingness_prompt(current_step, context_steps)

        try:
            response = await self.llm(prompt, response_format="json")
            score = float(response.get("interestingness_score", 0.5))

            # Penalize errors heavily
            if current_step.error or self.sanitizer.has_error(current_step.outputs):
                score = max(0.0, score - 0.4)

            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"    Warning: LLM call failed ({str(e)})")
            return 0.2 if current_step.error else 0.5

    def _build_interestingness_prompt(self, current_step, context_steps):
        """Build prompt for interestingness scoring."""
        # FIX 2: Sanitize context outputs
        context_text = ""
        if context_steps:
            context_lines = []
            for s in context_steps[-3:]:
                clean_output = self.sanitizer.sanitize(s.outputs, max_chars=100)
                context_lines.append(
                    f"- Step {s.step_number}: {s.tool_name} → {clean_output}"
                )
            context_text = "\n".join(context_lines)
        else:
            context_text = "This is the first step (no prior context)"

        # Sanitize current outputs
        clean_output = self.sanitizer.sanitize(current_step.outputs, max_chars=150)

        error_note = ""
        if current_step.error or self.sanitizer.has_error(current_step.outputs):
            error_note = "\n⚠️ WARNING: This step resulted in an ERROR. Heavily penalize interestingness."

        prompt = f"""Evaluate if this step is "interesting" for training data.

CONTEXT (Previous steps):
{context_text}

CURRENT STEP:
Tool: {current_step.tool_name}
Inputs: {json.dumps(current_step.inputs, indent=2)[:200]}
Outputs: {clean_output}
Success: {current_step.success}
{error_note}

SCORING CRITERIA:
HIGH (0.7-1.0): Introduces new entities, critical business logic, clear dependencies
MEDIUM (0.4-0.6): Routine operations, some context relevance
LOW (0.0-0.3): Trivial, errors, no clear purpose

OUTPUT (JSON):
{{
  "interestingness_score": 0.75,
  "reasoning": "Brief explanation"
}}

YOUR EVALUATION:"""
        return prompt

    def _filter_by_interestingness_and_dataflow(self, steps):
        """Filter steps by threshold + data flow."""
        if not steps: return []

        filtered = []
        last_interesting_step = None

        for step in steps:
            if step.interestingness_score >= self.threshold:
                if last_interesting_step is not None:
                    if self._check_data_flow(last_interesting_step, step):
                        filtered.append(step)
                        last_interesting_step = step
                else:
                    filtered.append(step)
                    last_interesting_step = step
        return filtered

    def _check_data_flow(self, step_a, step_b):
        """Check if there's data flow from step_a to step_b."""
        return len(step_a.get_output_entities() & step_b.get_input_entities()) > 0

    def _build_data_flow_graph(self, steps):
        """Build data flow graph."""
        flow = defaultdict(list)
        for i, step_a in enumerate(steps):
            outputs = step_a.get_output_entities()
            for step_b in steps[i+1:]:
                if outputs & step_b.get_input_entities():
                    flow[step_a.step_number].append(step_b.step_number)
        return flow

    def _infer_domain(self, steps):
        """Infer domain based on tool names and data."""
        domain_scores = defaultdict(int)
        for step in steps:
            step_text = json.dumps(step.inputs) + step.tool_name
            for domain, keywords in self.domain_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in step_text.lower():
                        domain_scores[domain] += 1
        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else "General"


# ============================================================================
# STEP 3: LOW-LEVEL TASK GENERATION (WITH FIXES)
# ============================================================================

class LowLevelTaskGenerator:
    """Generates rationales with cleaned context and no tool jargon."""

    def __init__(self, llm_caller):
        self.llm = llm_caller
        self.sanitizer = OutputSanitizer()

    async def generate_rationales(self, filtered_sequence: FilteredSequence) -> List[AnnotatedStep]:
        """Generate rationale for each step."""
        annotated_steps = []
        prereq_context = self._format_prerequisite_context(filtered_sequence.prerequisite_context)

        for step in filtered_sequence.steps:
            prompt = self._build_grounded_rationale_prompt(step, annotated_steps, prereq_context)

            try:
                response = await self.llm(prompt, response_format="text")
                rationale = response.strip() if isinstance(response, str) else response.get("rationale", "")
            except Exception:
                rationale = f"Perform {step.tool_name.replace('_', ' ')} operation with provided data."

            annotated_steps.append(AnnotatedStep(
                step_number=step.step_number,
                tool_name=step.tool_name,
                rationale=rationale,
                inputs=step.inputs,
                outputs=step.outputs,
                expected_output=step.outputs,
                entities_used=step.get_input_entities() | step.get_output_entities()
            ))
        return annotated_steps

    def _format_prerequisite_context(self, prereq_steps):
        """FIX 3: Format prerequisites cleanly (no failure claims)."""
        if not prereq_steps:
            return "No prior context - starting fresh."

        # Only successful steps should be here now (filtered upstream)
        context_lines = ["PREVIOUSLY COMPLETED:"]
        for step in prereq_steps[:5]:  # Limit to 5 most recent
            clean_output = self.sanitizer.sanitize(step.outputs, max_chars=60)
            context_lines.append(f"- {step.tool_name}: {clean_output}")
        return "\n".join(context_lines)

    def _build_grounded_rationale_prompt(self, step, prev_steps, context):
        """FIX 4: Remove tool jargon, focus on business action."""
        input_preview = json.dumps(step.inputs, indent=2)[:300]

        # Extract key input values for emphasis
        key_values = []
        try:
            for k, v in step.inputs.items():
                if v and isinstance(v, (str, int, float)) and str(v).strip():
                    key_values.append(f"{k}={v}")
        except: 
            pass
        key_values_str = ", ".join(key_values[:5])

        prompt = f"""You are an AI agent. Explain your NEXT action in natural language.

CONTEXT:
{context}

CURRENT ACTION:
Operation: {step.tool_name.replace('_', ' ').title()}
With Data: {key_values_str}

Full Inputs:
{input_preview}

RULES:
1. Write ONE sentence (first-person: "I am...", "I will...")
2. Use BUSINESS LANGUAGE - no tool names, no technical jargon
   - BAD: "I will execute owncloud_create_folder"
   - GOOD: "I am creating a new folder in cloud storage"
3. Reference SPECIFIC data values (IDs, names, paths)
4. Explain WHAT and WHY in plain terms

YOUR RATIONALE:"""
        return prompt


# ============================================================================
# STEP 4: HIGH-LEVEL TASK GENERATION (WITH FIXES)
# ============================================================================

class HighLevelTaskGenerator:
    """Synthesizes instruction with ground truth alignment."""

    def __init__(self, llm_caller):
        self.llm = llm_caller
        self.sanitizer = OutputSanitizer()

    async def generate_instruction(
        self, annotated_steps, prerequisite_context, domain, sequence_id
    ) -> Optional[SyntheticTask]:
        """Generate high-level instruction with validation."""

        # FIX 5: Check if final output has errors
        ground_truth = self._determine_task_ground_truth(annotated_steps)
        final_has_error = self.sanitizer.has_error(ground_truth.get("final_output"))

        # Decide: drop error tasks or convert to debug tasks
        if final_has_error:
            print(f"  ⚠ Skipping task {sequence_id}: final output contains error")
            return None  # Drop for now; could convert to debug task later

        # Collect all input data for grounding
        all_inputs = {}
        for step in annotated_steps:
            if step.inputs is None:
                continue
            for k, v in step.inputs.items():
                if v and str(v).strip():
                    all_inputs[f"{step.tool_name}.{k}"] = v

        prompt = self._build_story_instruction_prompt(annotated_steps, all_inputs, domain)

        try:
            response = await self.llm(prompt, response_format="json")
            instruction = response.get("instruction", "")
            success_criteria = response.get("success_criteria", [])
        except Exception:
            instruction = f"Execute {len(annotated_steps)}-step workflow in {domain}."
            success_criteria = ["All steps complete successfully"]

        # FIX 6: Align success criteria with ground truth
        if not success_criteria:
            success_criteria = self._generate_success_criteria_from_ground_truth(
                annotated_steps, ground_truth
            )

        task_id = f"task_{sequence_id}"
        prereq_data = [
            {"tool": s.tool_name, "outputs": self.sanitizer.sanitize(s.outputs, 100)}
            for s in prerequisite_context
        ]

        return SyntheticTask(
            task_id=task_id,
            instruction=instruction,
            chain_of_thought=annotated_steps,
            required_tools=list(set(s.tool_name for s in annotated_steps)),
            success_criteria=success_criteria,
            domain=domain,
            difficulty=self._assess_difficulty(annotated_steps),
            prerequisite_context=prereq_data,
            ground_truth=ground_truth,
            meta={
                "num_steps": len(annotated_steps),
                "num_prerequisites": len(prerequisite_context),
                "entry_tool": annotated_steps[0].tool_name,
                "exit_tool": annotated_steps[-1].tool_name
            }
        )

    def _determine_task_ground_truth(self, steps):
        """Determine ground truth with sanitized outputs."""
        if not steps:
            return {"final_output": None, "all_step_outputs": []}

        all_outputs = []
        for step in steps:
            all_outputs.append({
                "step_number": step.step_number,
                "tool": step.tool_name,
                "output": step.outputs
            })

        return {
            "final_output": steps[-1].outputs,
            "all_step_outputs": all_outputs,
            "final_entities": list(steps[-1].entities_used)[:20]
        }

    def _generate_success_criteria_from_ground_truth(self, steps, ground_truth):
        """Generate success criteria aligned with actual outputs."""
        criteria = []
        for step in steps:
            tool_action = step.tool_name.replace('_', ' ').title()
            # Extract a key value if possible
            key_val = ""
            for k, v in step.inputs.items():
                if isinstance(v, str) and v:
                    key_val = f" for {v}"
                    break
            criteria.append(f"{tool_action} completes successfully{key_val}")
        return criteria[:3]  # Top 3

    def _build_story_instruction_prompt(self, annotated_steps, all_inputs, domain):
        """Build prompt for natural instruction generation."""
        trace = []
        for s in annotated_steps:
            if s.inputs is None:
                inputs_str = "no inputs"
                continue
            inputs_str = ", ".join([f"{k}={v}" for k, v in list(s.inputs.items())[:3]])
            trace.append(f"Step {s.step_number}: {s.tool_name} ({inputs_str})")
        trace_text = "\n".join(trace)

        data_block = json.dumps(all_inputs, indent=2)[:500]

        prompt = f"""Create a natural user instruction from this execution trace.

DOMAIN: {domain}

TRACE (what actually happened):
{trace_text}

REQUIRED DATA (must appear in instruction):
{data_block}

RULES:
1. Write as a realistic business request (2-3 sentences)
2. Include ALL specific data values (IDs, names, paths, etc.)
3. Use business language, not technical terms
4. Do NOT say "use tool X" or "call API Y"
5. Be GROUNDED - if tools created data, ask to create; if retrieved, ask to retrieve

OUTPUT (JSON):
{{
  "instruction": "Full natural language request with all data",
  "success_criteria": [
    "Criterion 1",
    "Criterion 2"
  ]
}}

YOUR RESPONSE:"""
        return prompt

    def _assess_difficulty(self, steps):
        """Assess difficulty."""
        n = len(steps)
        if n <= 3: return "EASY"
        elif n <= 6: return "MEDIUM"
        else: return "HARD"


# ============================================================================
# PHASE 4: POST-PROCESSING (DEDUPE + DIVERSITY + VALIDATION)
# ============================================================================

class TaskPostProcessor:
    """
    Phase 4: Quality control after task generation.
    - Deduplication (exact + near-duplicate)
    - Diversity selection
    - Grounding validation
    """

    def __init__(self, diversity_ratio: float = 0.7):
        self.diversity_ratio = diversity_ratio
        self.sanitizer = OutputSanitizer()

    def process(self, tasks: List[SyntheticTask]) -> List[SyntheticTask]:
        """Main processing pipeline."""
        print(f"\n[PostProcessor] Starting with {len(tasks)} tasks")

        # Step 1: Validation (drop broken tasks)
        validated = self._validate_tasks(tasks)
        print(f"[PostProcessor] After validation: {len(validated)} tasks")

        # Step 2: Exact deduplication
        deduped = self._deduplicate_exact(validated)
        print(f"[PostProcessor] After exact dedupe: {len(deduped)} tasks")

        # Step 3: Near-duplicate removal (optional, can be expensive)
        # near_deduped = self._deduplicate_near(deduped)
        # print(f"[PostProcessor] After near dedupe: {len(near_deduped)} tasks")

        # Step 4: Diversity selection
        diverse = self._select_diverse(deduped)
        print(f"[PostProcessor] After diversity selection: {len(diverse)} tasks")

        return diverse

    def _validate_tasks(self, tasks):
        """FIX 7: Validate grounding and consistency."""
        validated = []
        for task in tasks:
            issues = []

            # Check 1: Success criteria vs ground truth
            gt_has_error = self.sanitizer.has_error(task.ground_truth.get("final_output"))
            success_claims_success = any(
                "success" in c.lower() or "complete" in c.lower() 
                for c in task.success_criteria
            )

            if gt_has_error and success_claims_success:
                issues.append("success_criteria contradicts error in ground_truth")

            # Check 2: Required inputs in instruction
            # Extract key input values
            required_values = set()
            for step in task.chain_of_thought:
                for k, v in step.inputs.items():
                    if isinstance(v, str) and len(v) > 3:
                        required_values.add(v)

            missing_values = [v for v in list(required_values)[:5] if v not in task.instruction]
            if len(missing_values) > len(required_values) * 0.5:
                issues.append(f"instruction missing key inputs: {missing_values[:2]}")

            # Check 3: Prerequisite sanity
            for prereq in task.prerequisite_context:
                if "Error:" in str(prereq.get("outputs", "")):
                    if "failed" not in task.instruction.lower():
                        issues.append("prerequisite error not reflected in instruction")

            if issues:
                print(f"  ✗ Dropping {task.task_id}: {'; '.join(issues)}")
            else:
                validated.append(task)

        return validated

    def _deduplicate_exact(self, tasks):
        """FIX 8: Remove exact duplicates by fingerprint."""
        seen_fingerprints = set()
        unique_tasks = []

        for task in tasks:
            fingerprint = self._compute_fingerprint(task)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_tasks.append(task)
            else:
                print(f"  ✗ Duplicate: {task.task_id}")

        return unique_tasks

    def _compute_fingerprint(self, task):
        """Compute stable fingerprint from task structure."""
        # Normalize inputs
        normalized_inputs = []
        for step in task.chain_of_thought:
            # Sort keys for stability
            sorted_inputs = {k: str(v) for k, v in sorted(step.inputs.items())}
            normalized_inputs.append((step.tool_name, json.dumps(sorted_inputs, sort_keys=True)))

        # Create fingerprint
        fingerprint_data = {
            "tools": tuple(task.required_tools),
            "steps": tuple(normalized_inputs),
            "num_steps": len(task.chain_of_thought)
        }

        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def _select_diverse(self, tasks):
        """FIX 9: Select diverse subset using greedy MMR-like approach."""
        if len(tasks) <= 10:
            return tasks  # Too few to filter

        # Target size
        target_size = max(10, int(len(tasks) * self.diversity_ratio))

        # Score each task by coverage (tools, domain, entities)
        task_features = []
        for task in tasks:
            features = set()
            features.add(f"domain:{task.domain}")
            features.update(f"tool:{t}" for t in task.required_tools)
            features.add(f"difficulty:{task.difficulty}")

            # Add entity diversity
            for step in task.chain_of_thought[:3]:  # First 3 steps
                features.update(f"entity:{e}" for e in list(step.entities_used)[:3])

            task_features.append((task, features))

        # Greedy selection: pick tasks that add most new features
        selected = []
        covered_features = set()

        # Start with task that has most features
        task_features.sort(key=lambda x: len(x[1]), reverse=True)

        for task, features in task_features:
            if len(selected) >= target_size:
                break

            # Calculate new features this task would add
            new_features = features - covered_features

            # If it adds value or we're below minimum, take it
            if len(new_features) > 0 or len(selected) < 5:
                selected.append(task)
                covered_features.update(features)

        print(f"  Selected {len(selected)} diverse tasks covering {len(covered_features)} features")
        return selected


# ============================================================================
# ORCHESTRATION PIPELINE
# ============================================================================

class CoTTaskSynthesisPipeline:
    """Complete pipeline with post-processing."""

    def __init__(self, llm_caller, enable_postprocessing: bool = True):
        self.sequence_extractor = IntelligentSequenceExtractor(llm_caller)
        self.low_level_gen = LowLevelTaskGenerator(llm_caller)
        self.high_level_gen = HighLevelTaskGenerator(llm_caller)
        self.post_processor = TaskPostProcessor(diversity_ratio=0.7)
        self.enable_postprocessing = enable_postprocessing

    async def process_trajectory(self, trajectory, max_tasks_per_cluster=3):
        """Process single trajectory."""
        print(f"[Pipeline] Processing trajectory: {trajectory['trajectory_id']}")

        # Extract sequences
        sequences = await self.sequence_extractor.extract_sequences(trajectory)
        print(f"[Pipeline] Extracted {len(sequences)} sequences")

        # Limit sequences
        sequences = sequences[:max_tasks_per_cluster * 3]

        all_tasks = []
        for seq in sequences:
            # Generate rationales
            annotated = await self.low_level_gen.generate_rationales(seq)

            # Generate instruction
            task = await self.high_level_gen.generate_instruction(
                annotated, seq.prerequisite_context, seq.domain, seq.sequence_id
            )

            if task:  # May be None if dropped due to errors
                all_tasks.append(task)
                # self.export_tasks(all_tasks, tasks_output_dir)
        print(f"[Pipeline] Generated {len(all_tasks)} tasks from trajectory")
        return all_tasks

    async def process_all_trajectories(self, trajectories):
        """Process multiple trajectories with global post-processing."""
        all_tasks = []

        for trajectory in trajectories:
            tasks = await self.process_trajectory(trajectory)
            all_tasks.extend(tasks)

        print(f"\n[Pipeline] Total tasks before post-processing: {len(all_tasks)}")

        # Phase 4: Post-processing
        if self.enable_postprocessing and all_tasks:
            all_tasks = self.post_processor.process(all_tasks)

        print(f"[Pipeline] Final task count: {len(all_tasks)}\n")
        return all_tasks

    def export_tasks(self, tasks, path):
        """Export tasks to JSON."""
        output = []
        for task in tasks:
            output.append({
                "task_id": task.task_id,
                "instruction": task.instruction,
                "prerequisite_context": task.prerequisite_context,
                "chain_of_thought": [
                    {
                        "step": step.step_number,
                        "rationale": step.rationale,
                        "tool": step.tool_name,
                        "inputs": step.inputs,
                        "expected_output": step.expected_output
                    }
                    for step in task.chain_of_thought
                ],
                "required_tools": task.required_tools,
                "success_criteria": task.success_criteria,
                "domain": task.domain,
                "difficulty": task.difficulty,
                "ground_truth": task.ground_truth,
                "meta": task.meta
            })

        with open(path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"[Pipeline] Exported {len(tasks)} tasks to {path}")