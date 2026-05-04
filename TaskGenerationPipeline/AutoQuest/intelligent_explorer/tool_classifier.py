"""
Tool Classification System
Classifies tools into CREATE, READ, UPDATE, DELETE, LIST operations
"""

import json
from typing import Dict, List, Any, Set
from enum import Enum
import json
from pathlib import Path
import asyncio


class ToolOperation(Enum):
    """Types of tool operations"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"
    UNKNOWN = "unknown"

environment = "arena"
class ToolClassifier:
    """
    Classifies tools based on their operation type and dependencies
    """
    
    CACHE_FILE = Path(f"./{environment}/cache/tool_classifications_grouped.json")
    DETAILED_CACHE_FILE = Path(f"./{environment}/cache/tool_classifications_detailed.json")
    
    def __init__(self, tools: List[Any], gpt_caller: callable):
        self.tools = tools
        self.gpt_caller = gpt_caller
        self.tool_classifications = {}
        self.resource_producers = {}
        self.resource_consumers = {}
    
    async def classify_all_tools(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Classify all tools by operation type and resource dependencies
        Uses cached classifications if available unless force_refresh=True
        """
        
        # Try to load from cache first
        if not force_refresh and self._load_from_cache():
            print("✅ Loaded tool classifications from cache")
            self._print_classification_summary()
            return self.tool_classifications
        
        print("🔍 Classifying tools by operation type...")
        
        # Batch classify for efficiency
        batch_size = 20
        for i in range(0, len(self.tools), batch_size):
            batch = self.tools[i:i+batch_size]
            await self._classify_batch(batch)
            print(f"   • Classified {min(i+batch_size, len(self.tools))}/{len(self.tools)} tools")
        
        # Analyze resource dependencies
        self._analyze_resource_dependencies()
        
        # Save to cache
        self._save_to_cache()
        
        self._print_classification_summary()
        
        return self.tool_classifications
    
    def _load_from_cache(self) -> bool:
        """Load classifications from cache file"""
        try:
            if not self.DETAILED_CACHE_FILE.exists():
                return False
            
            with open(self.DETAILED_CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
            
            # ✅ This reconstructs enums correctly
            self.tool_classifications = {}
            for tool_name, classification in cached_data['tool_classifications'].items():
                self.tool_classifications[tool_name] = {
                    "operation": ToolOperation[classification['operation']],  # ✅ String to Enum
                    "produces_resource": classification.get('produces_resource'),
                    "discovers_resource": classification.get('discovers_resource', []),
                    "requires_resources": classification.get('requires_resources', []),
                    "bootstrap_capable": classification.get('bootstrap_capable', False),
                    "reasoning": classification.get('reasoning', '')
                }
            
            self.resource_producers = cached_data.get('resource_producers', {})
            self.resource_consumers = cached_data.get('resource_consumers', {})
            
            return True
        except Exception as e:
            print(f"   ⚠️  Could not load cache: {e}")
            return False

    
    def _save_to_cache(self):
        """Save classifications to cache files in both formats"""
        try:
            # Create cache directory
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Format 1: Grouped by operation type (for quick lookup)
            grouped_classifications = {
                "CREATE": [],
                "READ": [],
                "UPDATE": [],
                "DELETE": [],
                "LIST": [],
                "SEARCH": [],
                "UNKNOWN": []
            }
            
            for tool_name, classification in self.tool_classifications.items():
                operation = classification['operation'].value.upper()
                grouped_classifications[operation].append({
                    "tool_name": tool_name,
                    "produces_resource": classification.get('produces_resource'),
                    "discovers_resource": classification.get('discovers_resource', []),
                    "requires_resources": classification.get('requires_resources', []),
                    "bootstrap_capable": classification.get('bootstrap_capable', False)
                })
            
            # Save grouped format
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(grouped_classifications, f, indent=2)
            
            print(f"💾 Saved grouped classifications to {self.CACHE_FILE}")
            
            # Format 2: Detailed format (for full reconstruction)
            detailed_data = {
                "tool_classifications": {
                    tool_name: {
                        "operation": classification['operation'].value.upper(),
                        "produces_resource": classification.get('produces_resource'),
                        "discovers_resource": classification.get('discovers_resource', []),
                        "requires_resources": classification.get('requires_resources', []),
                        "bootstrap_capable": classification.get('bootstrap_capable', False),
                        "reasoning": classification.get('reasoning', '')
                    }
                    for tool_name, classification in self.tool_classifications.items()
                },
                "resource_producers": self.resource_producers,
                "resource_consumers": self.resource_consumers,
                "metadata": {
                    "total_tools": len(self.tool_classifications),
                    "timestamp": str(Path(self.CACHE_FILE).stat().st_mtime) if self.CACHE_FILE.exists() else None
                }
            }
            
            # Save detailed format
            with open(self.DETAILED_CACHE_FILE, 'w') as f:
                json.dump(detailed_data, f, indent=2)
            
            print(f"💾 Saved detailed classifications to {self.DETAILED_CACHE_FILE}")
            
        except Exception as e:
            print(f"   ⚠️  Could not save cache: {e}")
    
    def _print_classification_summary(self):
        """Print summary of classifications"""
        print(f"✅ Tool classification summary:")
        print(f"   • CREATE operations: {sum(1 for t in self.tool_classifications.values() if t['operation'] == ToolOperation.CREATE)}")
        print(f"   • READ operations: {sum(1 for t in self.tool_classifications.values() if t['operation'] == ToolOperation.READ)}")
        print(f"   • UPDATE operations: {sum(1 for t in self.tool_classifications.values() if t['operation'] == ToolOperation.UPDATE)}")
        print(f"   • DELETE operations: {sum(1 for t in self.tool_classifications.values() if t['operation'] == ToolOperation.DELETE)}")
        print(f"   • LIST/SEARCH operations: {sum(1 for t in self.tool_classifications.values() if t['operation'] in [ToolOperation.LIST, ToolOperation.SEARCH])}\n")
    
    def get_tools_by_operation(self, operation: ToolOperation) -> List[str]:
        """Get all tools of a specific operation type"""
        return [
            tool_name for tool_name, classification in self.tool_classifications.items()
            if classification['operation'] == operation
        ]
    
    def export_grouped_classifications(self, output_file: str = None) -> Dict:
        """Export classifications grouped by operation type"""
        if output_file is None:
            output_file = self.CACHE_FILE
        
        grouped = {
            "CREATE": [],
            "READ": [],
            "UPDATE": [],
            "DELETE": [],
            "LIST": [],
            "SEARCH": [],
            "UNKNOWN": []
        }
        
        for tool_name, classification in self.tool_classifications.items():
            operation = classification['operation'].value.upper()
            grouped[operation].append({
                "tool_name": tool_name,
                "produces_resource": classification.get('produces_resource'),
                "discovers_resource": classification.get('discovers_resource', []),
                "requires_resources": classification.get('requires_resources', []),
                "bootstrap_capable": classification.get('bootstrap_capable', False)
            })
        
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(grouped, f, indent=2)
        
        return grouped
    
    async def _classify_batch(self, batch: List[Any]):
        """Classify a batch of tools with rate limit handling"""
        
        tools_info = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": self._extract_schema(tool)
        } for tool in batch]
        
        prompt = f"""
Analyze these tools and classify each one with PRECISE resource dependency tracking.

Tools:
{json.dumps(tools_info, indent=2)}

For each tool, determine:
1. **Operation type**: CREATE, READ, UPDATE, DELETE, LIST, SEARCH, or UNKNOWN

2. **produces_resource** (string or null):
   - What RESOURCE TYPE does this tool create? (e.g., "project", "issue", "user", "label")
   - ONLY for CREATE operations
   - Use singular form: "project" not "projects"
   - Match the domain entity: gitlab_create_repository → "project"
   
3. **discovers_resource** (string or null):
   - What RESOURCE TYPE does this tool discovers of search? (e.g., "project", "issue", "user", "label")
   - ONLY for SEARCH or LIST operations with no required arguments
   - Use singular form: "project" not "projects"
   - Match the domain entity: gitlab_search_all_repositories → "project"

4. **requires_resources** (array of RESOURCE TYPES, NOT parameter names):
   - What RESOURCE TYPES must exist before this tool can execute?
   - Use RESOURCE TYPES like ["project", "issue"], NOT parameter names like ["project_id", "title"]
   - Examples:
     * gitlab_create_issue → requires ["project"] (needs a project to exist)
     * gitlab_update_issue → requires ["project", "issue"] (needs both to exist)
     * gitlab_create_repository → requires [] (no dependencies)
     * gitlab_get_issue → requires ["project", "issue"]
   - For CREATE operations: list parent resource types needed
   - For READ/UPDATE/DELETE: list all resource types that must exist

5. **bootstrap_capable** (boolean):
   - true: Can execute without ANY existing resources (e.g., create_project, list_all)
   - false: Requires at least one resource to exist first

**Classification Rules**:
- CREATE: Creates new resources → produces_resource is NOT null
- READ: Retrieves by ID → requires specific resources
- UPDATE: Modifies existing → requires the resource being modified
- DELETE: Removes existing → requires the resource being deleted
- LIST: Lists without requiring specific IDs → often bootstrap_capable
- SEARCH: Search with filters → often bootstrap_capable

**CRITICAL**: 
- requires_resources MUST contain RESOURCE TYPES (project, issue, user)
- NOT parameter names (project_id, title, description)
- Think: "What resources must exist?" not "What parameters do I need?"

Return JSON:
{{
  "classifications": [
    {{
      "tool_name": "exact tool name",
      "operation": "CREATE|READ|UPDATE|DELETE|LIST|SEARCH|UNKNOWN",
      "produces_resource": "resource_type or null",
      "discovers_resource": "resource_type or null",
      "requires_resources": ["resource_type1", "resource_type2"],
      "bootstrap_capable": true/false,
      "reasoning": "brief explanation"
    }}
  ]
}}

Example:
{{
  "classifications": [
    {{
      "tool_name": "gitlab_create_repository",
      "operation": "CREATE",
      "produces_resource": "project",
      discovers_resource": null,
      "requires_resources": [],
      "bootstrap_capable": true,
      "reasoning": "Creates a new GitLab project without dependencies"
    }},
    {{
      "tool_name": "gitlab_create_issue",
      "operation": "CREATE",
      "produces_resource": "issue",
      discovers_resource": null,
      "requires_resources": ["project"],
      "bootstrap_capable": false,
      "reasoning": "Creates an issue, requires a project to exist first"
    }},
    {{
      "tool_name": "gitlab_update_issue",
      "operation": "UPDATE",
      "produces_resource": null,
      discovers_resource": null,
      "requires_resources": ["project", "issue"],
      "bootstrap_capable": false,
      "reasoning": "Updates an existing issue, requires both project and issue"
    }}
    {{
      "tool_name": "gitlab_search_all_repositories",
      "operation": "SEARCH",
      "produces_resource": null,
      discovers_resource": gitlab_repositories,
      "requires_resources": [],
      "bootstrap_capable": true,
      "reasoning": "Fetches all repositories without needing specific IDs"
    }}
  ]
}}
"""
        
        # Retry with exponential backoff for rate limits
        max_retries = 5
        for retry in range(max_retries):
            try:
                result = await self.gpt_caller(
                    prompt=prompt,
                    response_format="json",
                    model="gpt-4o",  # ✅ Use better model for accuracy
                    temperature=0.1  # ✅ Lower temperature for consistency
                )
                
                # Store classifications
                for classification in result.get("classifications", []):
                    tool_name = classification["tool_name"]
                    
                    op_str = classification["operation"]
                    operation = ToolOperation[op_str] if op_str in ToolOperation.__members__ else ToolOperation.UNKNOWN
                    
                    # ✅ Validate and clean requires_resources
                    requires_resources = classification.get("requires_resources", [])
                    requires_resources = self._clean_resource_requirements(requires_resources)
                    
                    self.tool_classifications[tool_name] = {
                        "operation": operation,
                        "produces_resource": classification.get("produces_resource"),
                        "discovers_resource": classification.get("discovers_resource"),
                        "requires_resources": requires_resources,
                        "bootstrap_capable": classification.get("bootstrap_capable", False),
                        "reasoning": classification.get("reasoning", "")
                    }
                
                return  # Success!
            
            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) * 30
                        print(f"  ⚠️  Rate limit hit, waiting {wait_time}s before retry {retry+1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"  ⚠️  Rate limit exhausted, using heuristic classification for batch")
                        self._heuristic_classification(batch)
                        return
                else:
                    print(f"  ✗ Classification error: {e}")
                    self._heuristic_classification(batch)
                    return
    
    def _extract_schema(self, tool: Any) -> Dict:
        """Extract schema from tool"""
        if hasattr(tool, 'args_schema'):
            schema = tool.args_schema
            if hasattr(schema, 'schema'):
                return schema.schema()
            return schema
        return {}
    
    def _clean_resource_requirements(self, requires_resources: List[str]) -> List[str]:
        """
        Clean and validate resource requirements
        Remove parameter suffixes to get resource types
        """
        cleaned = []
        for req in requires_resources:
            # Remove common suffixes to get base resource type
            resource_type = req.replace("_id", "").replace("_iid", "").replace("_name", "")
            resource_type = resource_type.replace("Id", "").replace("Name", "")
            resource_type = resource_type.strip().lower()
            
            # Skip common non-resource parameters
            if resource_type in ["title", "description", "name", "content", "message", "data"]:
                continue
            
            if resource_type and resource_type not in cleaned:
                cleaned.append(resource_type)
        
        return cleaned

    def _heuristic_classification(self, batch: List[Any]):
        """Fallback heuristic classification when LLM fails"""
        for tool in batch:
            name_lower = tool.name.lower()
            
            # Extract potential resource type from name
            resource_type = None
            for resource in ['project', 'repository', 'user', 'issue', 'merge_request', 
                           'email', 'calendar', 'ticket', 'label', 'milestone', 'branch']:
                if resource in name_lower:
                    resource_type = resource
                    break
            
            # Classify operation
            if any(word in name_lower for word in ['create', 'add', 'new', 'register']):
                operation = ToolOperation.CREATE
                bootstrap = resource_type is None
                discovers = None
                produces = resource_type
                requires = []
            elif any(word in name_lower for word in ['list', 'all']):
                operation = ToolOperation.LIST
                bootstrap = True
                discovers = resource_type
                produces = None
                requires = []
            elif any(word in name_lower for word in ['search', 'find', 'query']):
                operation = ToolOperation.SEARCH
                bootstrap = True
                discovers = resource_type
                produces = None
                requires = []
            elif any(word in name_lower for word in ['get', 'fetch', 'retrieve', 'read']):
                operation = ToolOperation.READ
                bootstrap = False
                discovers = resource_type
                produces = None
                requires = [resource_type] if resource_type else []
            elif any(word in name_lower for word in ['update', 'edit', 'modify', 'patch']):
                operation = ToolOperation.UPDATE
                bootstrap = False
                discovers = None
                produces = None
                requires = [resource_type] if resource_type else []
            elif any(word in name_lower for word in ['delete', 'remove', 'destroy']):
                operation = ToolOperation.DELETE
                bootstrap = False
                discovers = None
                produces = None
                requires = [resource_type] if resource_type else []
            else:
                operation = ToolOperation.UNKNOWN
                bootstrap = True
                produces = None
                discovers = None
                requires = []
            
            self.tool_classifications[tool.name] = {
                "operation": operation,
                "produces_resource": produces,
                "discovers_resource": discovers,
                "requires_resources": requires,
                "bootstrap_capable": bootstrap,
                "reasoning": "Heuristic classification (LLM unavailable)"
            }
    
    def _analyze_resource_dependencies(self):
        """Analyze which tools produce/consume which resources"""
        
        # Build producer map
        for tool_name, classification in self.tool_classifications.items():
            if classification["produces_resource"]:
                resource_type = classification["produces_resource"]
                if resource_type not in self.resource_producers:
                    self.resource_producers[resource_type] = []
                self.resource_producers[resource_type].append(tool_name)
        
        # Build consumer map
        for tool_name, classification in self.tool_classifications.items():
            for required_resource in classification["requires_resources"]:
                if required_resource not in self.resource_consumers:
                    self.resource_consumers[required_resource] = []
                self.resource_consumers[required_resource].append(tool_name)
    
    def get_bootstrap_tools(self) -> List[str]:
        """Get tools that can work without prerequisites"""
        return [
            tool_name for tool_name, classification in self.tool_classifications.items()
            if classification["bootstrap_capable"] or classification["operation"] in [ToolOperation.CREATE, ToolOperation.LIST]
        ]
    
    def get_producer_tools_for_resource(self, resource_type: str) -> List[str]:
        """Get tools that produce a given resource type"""
        return self.resource_producers.get(resource_type, [])
    
    def get_required_resources(self, tool_name: str) -> List[str]:
        """Get resources required by a tool"""
        classification = self.tool_classifications.get(tool_name, {})
        return classification.get("requires_resources", [])
    
    def can_execute_without_prerequisites(self, tool_name: str, available_resources: Dict[str, Any]) -> bool:
        """Check if a tool can execute given available resources"""
        required = self.get_required_resources(tool_name)
        
        if not required:
            return True
        
        # Check if all required resources are available
        for resource_type in required:
            if resource_type not in available_resources or not available_resources[resource_type]:
                return False
        
        return True
