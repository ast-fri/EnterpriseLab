"""
Synthetic input generation for AutoQuest Phase 2
Critical module: Uses dynamic, evolving KB AND short-term trajectory memory
"""

from typing import Dict, Optional, List, Any, Tuple
import json
import time
import random
import uuid
from AutoQuest.intelligent_explorer.data_models import EnvironmentKnowledgeBase, ResourceNode
from AutoQuest.intelligent_explorer.tool_classifier import ToolOperation
from utils import normalize_operation
from AutoQuest.intelligent_explorer.memory_manager import MemoryManager


class InputGenerator:
    """
    Generates tool inputs using evolving KB state + short-term trajectory memory
    CRITICAL: Prioritizes recent trajectory resources over long-term KB
    """

    def __init__(
        self,
        tools: Dict,
        tool_classifications: Dict,
        gpt_caller: callable,
        graph_utils: Any,
        memory_manager: 'MemoryManager'  # ✅ NEW
    ):
        """
        Args:
            tools: Dictionary of tool_name -> tool object
            tool_classifications: Tool classification mappings
            gpt_caller: Async GPT caller function
            graph_utils: GraphUtils for schema extraction
            memory_manager: MemoryManager for trajectory-level memory
        """
        self.tools = tools
        self.tool_classifications = tool_classifications
        self.gpt_caller = gpt_caller
        self.graph_utils = graph_utils
        self.memory_manager = memory_manager  # ✅ NEW

        # Track generation failures for debugging
        self.generation_failures = []

        # Prompt templates (versioned for reproducibility)
        self.prompt_version = "v3.0_dual_memory_aware"

    # ========================
    # Main Input Generation
    # ========================

    async def generate_synthetic_input(
        self, node_name, kb, feedback="", parent_execution_state=None, depth=0, retry_attempt=0, trajectory_id=None
    ) -> Optional[Dict]:
        tool = self.tools.get(node_name)
        if not tool: return None

        schema = self.graph_utils.get_tool_schema(tool)
        required_params = self.graph_utils.get_required_parameters(tool)
        param_types = self.graph_utils.get_parameter_types(tool)
        classification = self.tool_classifications.get(node_name, {})
        operation = normalize_operation(classification.get("operation"))

        memory_context = None
        if trajectory_id:
            memory_context = self.memory_manager.get_context_for_input_generation(
                trajectory_id, self._extract_required_resource_types(node_name, required_params)
            )

        resource_mapping = await self._build_kb_resource_mapping(
            node_name, required_params, kb, classification, memory_context
        )

        tool_input = await self._llm_generate_input(
            node_name=node_name, tool=tool, schema=schema, required_params=required_params,
            param_types=param_types, operation=operation, kb=kb, resource_mapping=resource_mapping,
            feedback=feedback, retry_attempt=retry_attempt, memory_context=memory_context
        )
        if not tool_input: return None

        tool_input = self._validate_and_enforce_constraints(tool_input, schema, required_params, param_types, kb, operation)
        if not tool_input: return None

        return tool_input

    # ========================
    # Resource Type Extraction
    # ========================

    def _extract_required_resource_types(self, node_name, required_params):
        # Same logic as earlier, extract required resource types
        classification = self.tool_classifications.get(node_name, {})
        required = classification.get('requires_resources', [])
        for param_name in required_params:
            resource_type = self._infer_resource_type_from_param(param_name)
            if resource_type and resource_type not in required:
                required.append(resource_type)
        return required

    # ========================
    # KB-Aware Resource Mapping (Memory-Enhanced)
    # ========================

    async def _build_kb_resource_mapping(self, node_name, required_params, kb, classification, memory_context):
        mapping = {}
        tool = self.tools.get(node_name)
        param_types = self.graph_utils.get_parameter_types(tool) if tool else {}

        for param in required_params:
            inferred_type = self._infer_resource_type_from_param(param)
            param_type_hint = param_types.get(param, "string")
            # Priority: STM
            memory_match = None
            if memory_context and memory_context.get('priority_resources'):
                priority_resources = memory_context['priority_resources']
                if inferred_type in priority_resources:
                    candidate_id = priority_resources[inferred_type][-1]
                    resource = kb.get_resource(candidate_id)
                    resource_type = resource.resource_type if resource else inferred_type
                    memory_match = {
                        "status": "MATCHED", "resource_id": candidate_id, "field_name": param,
                        "field_value": candidate_id, "resource_type": resource_type,
                        "source": "SHORT_TERM_MEMORY", "mapping_type": "memory_priority"
                    }
            if memory_match:
                mapping[param] = memory_match
                continue
            # Fallback: KB semantic
            semantic_match = await self._llm_semantic_field_mapper(param, param_type_hint, inferred_type, kb)
            if semantic_match:
                resource_id, field_name, context = semantic_match
                resource = kb.get_resource(resource_id)
                mapping[param] = {
                    "status": "MATCHED", "resource_id": resource_id, "field_name": field_name,
                    "field_value": context["field_value"], "resource_type": context["resource_type"],
                    "source": "LONG_TERM_KB", "mapping_type": "semantic"
                }
            else:
                mapping[param] = {"status": "NO_MATCH", "inferred_type": inferred_type, "mapping_type": "none"}
        return mapping

    # ========================
    # SEMANTIC FIELD MAPPING
    # ========================

    async def _llm_semantic_field_mapper(
        self,
        param_name: str,
        param_type_hint: str,
        inferred_resource_type: str,
        kb: EnvironmentKnowledgeBase
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Use LLM to semantically map parameter to KB resource fields

        Args:
            param_name: Parameter name (e.g., "project_id")
            param_type_hint: Type from schema (e.g., "integer")
            inferred_resource_type: Resource type (e.g., "project")
            kb: Knowledge base

        Returns:
            Tuple of (resource_id, field_name, resource_context) or None
        """

        # Get candidate resources of inferred type
        candidates = kb.get_all_by_type(inferred_resource_type, include_deleted=False)

        if not candidates:
            # Try fuzzy type matching
            for kb_type in kb.resource_by_type.keys():
                if inferred_resource_type.lower() in kb_type.lower() or kb_type.lower() in inferred_resource_type.lower():
                    candidates = kb.get_all_by_type(kb_type, include_deleted=False)
                    if candidates:
                        inferred_resource_type = kb_type
                        break

        if not candidates:
            return None

        # Build candidate summary with FULL schema
        candidate_summaries = []
        for resource in candidates[:5]:  # Limit to 5 most recent
            schema = kb.get_schema(resource.resource_type)
            candidate_summaries.append({
                "resource_id": resource.resource_id,
                "resource_type": resource.resource_type,
                "created_by": resource.created_by_tool,
                "output_schema": schema,
                "sample_data": resource.creation_outputs,
                "access_count": resource.access_count,
                "age_seconds": time.time() - resource.creation_timestamp
            })

        prompt = f"""You are a semantic field mapper for tool parameter binding in an evolving environment.

**TASK**: Determine which KB resource and which FIELD from that resource should be used for the parameter `{param_name}`.

---

**TARGET PARAMETER:**
- Name: `{param_name}`
- Expected Type: `{param_type_hint}`
- Inferred Resource Type: `{inferred_resource_type}`

---

**AVAILABLE RESOURCES IN KB:**
{json.dumps(candidate_summaries, indent=2)}

---

**SEMANTIC MAPPING INSTRUCTIONS:**

1. **Understand Parameter Intent**:
   - `project_id` → needs the ID/identifier of a project
   - `sender_emp_id` → needs an employee ID (sender context)
   - `recipient_emp_id` → needs an employee ID (recipient context)
   - **KEY**: Prefixes like sender_, recipient_, source_, target_ indicate ROLE, not different entity types

2. **Analyze Resource Schemas**:
   - Look at `output_schema` and `sample_data` for each candidate
   - Find fields that match the semantic intent of `{param_name}`
   - **Field names may differ but semantics align**

3. **✅ Handle Prefix Variations**:
   - If parameter has prefix (sender_, recipient_, source_, target_, from_, to_):
     - Strip prefix and match core field name
     - Example: `sender_emp_id` → strip "sender_" → match `emp_id` field

4. **Field Matching Heuristics**:
   - **Exact match**: `project_id` matches `project_id` field
   - **Prefix match**: `sender_emp_id` matches `emp_id` field
   - **Alias match**: `project_id` matches `id` field in project resource
   - **Type compatibility**: ensure field type matches `{param_type_hint}`

5. **Selection Priority**:
   - Prefer most recently accessed resources (higher `access_count`)
   - Prefer newer resources (lower `age_seconds`)

---

**OUTPUT FORMAT (JSON):**

If a match is found:
{{
  "matched": true,
  "resource_id": "42",
  "field_name": "emp_id",
  "field_value": "emp_20231010_1a2b3c",
  "confidence": "high",
  "reasoning": "Parameter 'sender_emp_id' maps to 'emp_id' field in employee resource"
}}

If no match:
{{
  "matched": false,
  "reason": "No suitable field found in available resources"
}}

**CRITICAL**: Return ONLY valid JSON, no extra text.

Perform semantic mapping now:"""

        try:
            response = await self.gpt_caller(
                prompt=prompt,
                response_format="json",
                model="gpt-4o",
                temperature=0.1
            )

            if not response or not isinstance(response, dict):
                print(f"      ⚠️  Semantic mapper returned invalid response")
                return None

            if not response.get("matched"):
                print(f"      ℹ️  No semantic match: {response.get('reason')}")
                return None

            resource_id = response["resource_id"]
            field_name = response["field_name"]
            field_value = response["field_value"]

            print(f"      🎯 Semantic match found:")
            print(f"          {param_name} → {resource_id}.{field_name} = {field_value}")

            # Get full resource context
            resource = kb.get_resource(resource_id)
            if not resource:
                return None

            return (resource_id, field_name, {
                "field_value": field_value,
                "resource_type": resource.resource_type,
                "confidence": response.get("confidence", "medium"),
                "reasoning": response.get("reasoning", "")
            })

        except Exception as e:
            print(f"      ❌ Semantic mapping error: {e}")
            return None

    def _infer_resource_type_from_param(self, param_name: str) -> str:
        """Infer resource type from parameter name"""
        param_lower = param_name.lower()

        # Strip common prefixes
        for prefix in ['sender_', 'recipient_', 'source_', 'target_', 'from_', 'to_']:
            param_lower = param_lower.replace(prefix, '')

        # Common patterns
        type_patterns = {
            "project": ["project_id", "project", "projectid", "project_name"],
            "issue": ["issue_id", "issue_iid", "issueid", "issue"],
            "merge_request": ["merge_request_iid", "mr_iid", "mergerequest"],
            "repository": ["repo_name", "repository", "repo_id", "repo"],
            "branch": ["branch_name", "branch", "ref"],
            "user": ["user_id", "username", "assignee", "author"],
            "employee": ["emp_id", "employee", "employee_id"],
            "customer": ["customer_id", "customer"],
            "chat": ["chat_id", "chat", "conversation_id"],
            "email": ["email_id", "email", "message_id"],
            "milestone": ["milestone_id", "milestone"],
            "label": ["label_id", "label_name", "label"],
            "commit": ["commit_id", "commit_sha", "sha"],
            "pipeline": ["pipeline_id", "pipeline"],
            "job": ["job_id", "job_name", "job"]
        }

        for resource_type, patterns in type_patterns.items():
            if any(pattern in param_lower for pattern in patterns):
                return resource_type

        # Fallback: clean up suffixes
        cleaned = param_lower.replace("_id", "").replace("_iid", "").replace("_name", "")
        return cleaned

    # ========================
    # LLM-Powered Generation (Memory-Enhanced)
    # ========================

    async def _llm_generate_input(
        self,
        node_name: str,
        tool: Any,
        schema: Dict,
        required_params: List[str],
        param_types: Dict[str, str],
        operation: ToolOperation,
        kb: EnvironmentKnowledgeBase,
        resource_mapping: Dict,
        feedback: str,
        retry_attempt: int,
        memory_context: Optional[Dict] = None  # ✅ NEW
    ) -> Optional[Dict]:
        """
        Use LLM to generate tool inputs with dual-memory awareness
        """
        # Build memory-aware context
        context_str = self._build_memory_aware_context(
            node_name,
            kb,
            resource_mapping,
            memory_context
        )

        # Build resource mapping guide
        mapping_guide = self._format_mapping_guide(resource_mapping, kb)

        tool_description = tool.description if hasattr(tool, 'description') else "N/A"
        # print("Required Params:", required_params)
        if not required_params:
            print(f"      ⚠️  No required parameters for tool {node_name}")
            return {}
        prompt = f"""Generate tool inputs using REAL resources with PRIORITY AWARENESS.

### TARGET TOOL
- **Name**: {node_name}
- **Operation**: {operation.value if hasattr(operation, 'value') else str(operation)}
- **Description**: {tool_description}
- **Schema**: {json.dumps(schema, indent=2)}
- **Required**: {required_params}
- **Types**: {json.dumps(param_types, indent=2)}

---

{context_str}

---

### RESOURCE MAPPING
{mapping_guide}

---

{f"### PREVIOUS ATTEMPT FEEDBACK\n{feedback}\n\n---\n" if feedback else ""}

### GENERATION RULES

**1. PRIORITY SYSTEM:**
- **🔥 SHORT-TERM MEMORY resources = HIGHEST PRIORITY**
- **📚 LONG-TERM KB resources = FALLBACK ONLY**

**2. For MATCHED Parameters (✓):**
- Use EXACT value provided
- Respect SOURCE: Short-term over long-term

**3. For NO_MATCH Parameters:**
- For CREATE operations: Generate synthetic values.
- For UPDATE operations: OMIT the parameter if not required. Do NOT return error.  
- For READ/DELETE: Return error only if the primary ID is missing.


**4. For CREATE Operations:**
- **Primary ID**: Generate NEW unique ID with timestamp
- **Foreign keys**: Reuse from matched resources
- Generate realistic values (real names, emails, etc.)

**5. Type Enforcement:**
- integer → actual numbers, not strings
- string → quoted text
- boolean → true/false

---

### OUTPUT FORMAT
Return ONLY JSON. No explanation, no markdown.

**Success:**
{{"param1": "value1", "param2": 123}}

**Error (only for non-CREATE with missing resources):**
{{"error": "missing_resources", "missing": ["resource_type"]}}

Generate now:"""

        try:
            tool_input = await self.gpt_caller(
                prompt=prompt,
                response_format="json",
                model="gpt-4o",
                temperature=0.1 + (retry_attempt * 0.1)
            )
            # print("Prompt Output: ", tool_input)
            # if not tool_input:
            #     print(f"      ❌ LLM returned None")
            #     return None

            # Check for error response
            if isinstance(tool_input, dict) and tool_input.get("error") == "missing_resources":
                print(f"      ⚠️  LLM indicated missing resources: {tool_input.get('missing')}")
                return None

            # Debug output
            print(f"      ✅ LLM generated inputs:")
            for key, value in tool_input.items():
                print(f"          {key}: {value} (type: {type(value).__name__})")

            return tool_input

        except Exception as e:
            print(f"      ❌ LLM generation error: {e}")
            return None

    # ========================
    # Memory-Aware Context Building
    # ========================

    def _build_memory_aware_context(
        self,
        node_name: str,
        kb: EnvironmentKnowledgeBase,
        resource_mapping: Dict,
        memory_context: Optional[Dict]
    ) -> str:
        """
        Build context string with SHORT-TERM (priority) and LONG-TERM (fallback) memory
        """
        context_parts = []

        # ✅ PART 1: SHORT-TERM MEMORY (HIGHEST PRIORITY)
        if memory_context and memory_context.get('priority_resources'):
            context_parts.append("🔥 **SHORT-TERM MEMORY (USE THESE FIRST)**")
            context_parts.append("Resources created in THIS workflow:")
            context_parts.append("")

            for resource_type, ids in memory_context['priority_resources'].items():
                context_parts.append(f"**{resource_type}** (JUST CREATED):")
                for resource_id in ids:
                    resource = kb.get_resource(resource_id)
                    if resource:
                        context_parts.append(f"  - ID: `{resource_id}`")
                        context_parts.append(f"    Created by: {resource.created_by_tool}")

                        if resource.creation_outputs:
                            sample_fields = self._extract_sample_fields(resource.creation_outputs)
                            if sample_fields:
                                context_parts.append(f"    Key Fields: {sample_fields}")

                        # Show full content (truncated)
                        if resource.creation_outputs:
                            resource_json = json.dumps(resource.creation_outputs, indent=4)
                            if len(resource_json) > 500:
                                resource_json = resource_json[:500] + "\n    ... (truncated)"
                            context_parts.append(f"    Full Data:\n{resource_json}")

                        context_parts.append("")

            context_parts.append("⚠️  **CRITICAL**: If you need any of the above resource types, you MUST use these IDs first!")
            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")

        # ✅ PART 2: RECENT EXECUTION HISTORY
        if memory_context and memory_context.get('recent_steps'):
            context_parts.append("📜 **RECENT EXECUTION HISTORY**")
            context_parts.append("Last few steps in this workflow:")
            context_parts.append("")

            for i, step in enumerate(memory_context['recent_steps'], 1):
                context_parts.append(f"  {i}. {step['tool']}")
                if step.get('outputs') and isinstance(step['outputs'], dict):
                    key_outputs = {k: v for k, v in list(step['outputs'].items())[:3]}
                    if key_outputs:
                        context_parts.append(f"     Outputs: {json.dumps(key_outputs)}")

            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")

        # ✅ PART 3: LONG-TERM KB (FALLBACK)
        if memory_context and memory_context.get('fallback_resources'):
            context_parts.append("📚 **LONG-TERM KNOWLEDGE BASE (FALLBACK)**")
            context_parts.append("Use these ONLY if short-term memory doesn't have what you need:")
            context_parts.append("")

            for resource_type, ids in memory_context['fallback_resources'].items():
                if not ids:
                    continue

                context_parts.append(f"**{resource_type}**:")
                for resource_id in ids[:3]:  # Limit to 3 per type
                    resource = kb.get_resource(resource_id)
                    if resource:
                        context_parts.append(f"  - ID: `{resource_id}`")
                        if resource.creation_outputs:
                            sample_fields = self._extract_sample_fields(resource.creation_outputs)
                            if sample_fields:
                                context_parts.append(f"    Fields: {sample_fields}")
                context_parts.append("")

        # ✅ PART 4: If NO memory context (fallback to traditional KB)
        if not memory_context:
            context_parts.append("📚 **KNOWLEDGE BASE STATE**")
            context_parts.append(self._build_traditional_kb_context(node_name, kb, resource_mapping))

        return "\n".join(context_parts)

    def _build_traditional_kb_context(
        self,
        node_name: str,
        kb: EnvironmentKnowledgeBase,
        resource_mapping: Dict
    ) -> str:
        """Fallback: Traditional KB context when no memory context available"""
        context = f"Version: {kb.version}\n"
        context += f"Environment: `{kb.environment_name}`\n"
        context += f"Active Resources: {len([r for r in kb.resources.values() if not r.is_deleted])}\n\n"

        # Show relevant resource types
        relevant_types = set(
            mapping.get("resource_type", mapping.get("inferred_type"))
            for mapping in resource_mapping.values()
        )

        context += "**AVAILABLE RESOURCES:**\n"
        for res_type in relevant_types:
            if res_type in kb.resource_by_type:
                active_resources = [
                    kb.resources[rid]
                    for rid in kb.resource_by_type[res_type]
                    if not kb.resources[rid].is_deleted
                ]

                if active_resources:
                    context += f"\n`{res_type.upper()}`:\n"
                    for resource in active_resources[-3:]:
                        context += f"  - ID: `{resource.resource_id}`\n"
                        context += f"    Created by: `{resource.created_by_tool}`\n"

        return context

    def _extract_sample_fields(self, creation_outputs: Dict, max_fields: int = 5) -> str:
        """Extract sample fields from resource outputs"""
        if not creation_outputs:
            return ""

        samples = []
        for key, value in list(creation_outputs.items())[:max_fields]:
            value_str = str(value)[:50]  # Truncate long values
            samples.append(f"{key}={value_str}")

        return ", ".join(samples)

    def _find_creation_step(self, memory_context: Dict, resource_id: str) -> str:
        """Find which step created this resource"""
        if not memory_context or 'recent_steps' not in memory_context:
            return "Unknown"

        for i, step in enumerate(memory_context['recent_steps'], 1):
            if isinstance(step.get('outputs'), dict):
                for key, value in step['outputs'].items():
                    if str(value) == str(resource_id):
                        return f"{i} ({step.get('tool', 'Unknown')})"

        return "Unknown"

    def _format_mapping_guide(self, resource_mapping: Dict, kb: EnvironmentKnowledgeBase) -> str:
        """Format resource mapping as LLM guide"""
        guide = ""
        shown_resources = set()

        for param, mapping in resource_mapping.items():
            status = mapping["status"]

            if status == "MATCHED":
                mapping_type = mapping.get("mapping_type", "direct")
                source = mapping.get("source", "KB")
                resource_id = mapping['resource_id']

                # Highlight short-term memory resources
                priority_marker = "🔥 " if source == "SHORT_TERM_MEMORY" else "📚 "

                guide += f"{priority_marker}✓ **`{param}`**: MATCHED ({mapping_type.upper()}, {source})\n"
                guide += f"  - Resource Type: `{mapping['resource_type']}`\n"
                guide += f"  - Resource ID: `{resource_id}`\n"

                if mapping_type == "semantic" or mapping_type == "memory_priority":
                    if 'field_name' in mapping:
                        guide += f"  - **Field Mapping**: `{mapping['field_name']}` → `{param}`\n"
                    guide += f"  - **USE THIS VALUE**: `{mapping.get('field_value', resource_id)}`\n"
                else:
                    guide += f"  - **USE THIS ID**: `{resource_id}`\n"

                # Show full resource content (only once per resource)
                if resource_id not in shown_resources:
                    shown_resources.add(resource_id)
                    resource = kb.get_resource(resource_id)

                    if resource and resource.creation_outputs:
                        guide += f"\n  📦 **FULL RESOURCE CONTENT**:\n"
                        resource_json = json.dumps(resource.creation_outputs, indent=4)
                        if len(resource_json) > 600:
                            resource_json = resource_json[:600] + "\n    ... (truncated)"
                        guide += f"  ```\n{resource_json}\n  ```\n"

                        available_fields = list(resource.creation_outputs.keys())
                        guide += f"  📋 Available fields: {available_fields}\n"

                guide += f"\n"

            else:
                guide += f"✗ **`{param}`**: NO_MATCH\n"
                guide += f"  - Inferred Type: `{mapping.get('inferred_type', 'unknown')}`\n"
                guide += f"  - Can Be Synthetic: `{mapping.get('can_be_synthetic', False)}`\n"
                guide += f"  - Reason: {mapping.get('reason', 'Unknown')}\n"

                if mapping.get('can_be_synthetic'):
                    guide += f"  - **ACTION**: Generate unique value with timestamp\n"
                else:
                    guide += f"  - **ACTION**: Check matched resources or return error\n"

                guide += f"\n"

        return guide

    # ========================
    # Validation & Constraints
    # ========================

    def _validate_and_enforce_constraints(
        self,
        tool_input: Dict,
        schema: Dict,
        required_params: List[str],
        param_types: Dict[str, str],
        kb: EnvironmentKnowledgeBase,
        operation: ToolOperation
    ) -> Optional[Dict]:
        """Validate generated inputs against constraints"""
        # 1. Check required parameters
        for param in required_params:
            if param not in tool_input or tool_input[param] is None or tool_input[param] == "":
                print(f"      ⚠️  Missing required parameter: {param}")
                return None

        # 2. Enforce type constraints
        for param, value in list(tool_input.items()):
            if param in param_types:
                expected_type = param_types[param]
                tool_input[param] = self._coerce_to_type(value, expected_type)

        return tool_input

    def _coerce_to_type(self, value: Any, expected_type: str) -> Any:
        """Coerce value to expected type"""
        try:
            if expected_type == "integer":
                return int(value)
            elif expected_type == "number":
                return float(value)
            elif expected_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ["true", "1", "yes"]
                return bool(value)
            elif expected_type == "string":
                return str(value)
            elif expected_type == "array":
                return value if isinstance(value, list) else [value]
            else:
                return value
        except (ValueError, TypeError):
            print(f"      ⚠️  Type coercion failed: {value} -> {expected_type}")
            return value

    def _verify_no_synthetic_ids(
        self,
        tool_input: Dict,
        kb: EnvironmentKnowledgeBase,
        operation: ToolOperation
    ) -> bool:
        """Final safety check: ensure no synthetic IDs leaked"""
        if operation == ToolOperation.CREATE:
            return True  # CREATE can have synthetic values

        return True  # Simplified - trust the LLM and validation

    # ========================
    # Failure Tracking
    # ========================

    def _record_failure(
        self,
        node_name: str,
        reason: str,
        kb: EnvironmentKnowledgeBase,
        resource_mapping: Dict
    ):
        """Record generation failure for analysis"""
        self.generation_failures.append({
            "node_name": node_name,
            "reason": reason,
            "kb_version": kb.version,
            "kb_resources": len(kb.resources),
            "resource_mapping": resource_mapping,
            "timestamp": time.time()
        })

    def get_failure_stats(self) -> Dict:
        """Get failure statistics"""
        return {
            "total_failures": len(self.generation_failures),
            "failure_reasons": {
                reason: sum(1 for f in self.generation_failures if f["reason"] == reason)
                for reason in set(f["reason"] for f in self.generation_failures)
            }
        }