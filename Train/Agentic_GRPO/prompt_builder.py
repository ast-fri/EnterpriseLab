"""
PromptBuilder - Builds ReAct-style system prompts from tool descriptions.

FIXED: 
- Uses proper tool schema instead of bound methods
- Examples use real EnterpriseBench tool names
- Clear format requirements, no XML tags allowed
"""

import json
from typing import List, Dict, Any, Union


class PromptBuilder:
    """
    Builds ReAct-style system prompts with strict format requirements.
    
    FIXED: Works with tool schemas, not bound methods
    """

    def __init__(self, tool_methods: Union[List[Dict[str, Any]], List[str], Dict[str, Any]]):
        """
        Args:
            tool_methods: Tools in various formats:
                - Dict[str, Dict]: {tool_name: {description: ..., args_schema: ...}}
                - List[Dict]: [{name: ..., description: ..., args_schema: ...}]
                - List[str]: [tool_name1, tool_name2, ...]
        """
        # Normalize to list of dicts with proper schema
        self.tool_methods = self._normalize_tools(tool_methods)

    def _normalize_tools(self, tool_methods: Any) -> List[Dict[str, Any]]:
        """
        FIXED: Normalize various tool formats to standard schema.
        
        Returns:
            List of dicts with: name, description, args_schema
        """
        normalized = []
        
        if isinstance(tool_methods, dict):
            # Dict format: {tool_name: {description: ..., args_schema: ...}}
            for name, info in tool_methods.items():
                if isinstance(info, dict):
                    tool_dict = {'name': name}
                    
                    # Extract description
                    if 'description' in info:
                        tool_dict['description'] = info['description']
                    else:
                        tool_dict['description'] = f"Tool: {name}"
                    
                    # Extract args schema
                    if 'args_schema' in info:
                        tool_dict['args_schema'] = info['args_schema']
                    else:
                        tool_dict['args_schema'] = {}
                    
                    normalized.append(tool_dict)
                elif isinstance(info, str):
                    # Simple string description
                    normalized.append({
                        'name': name,
                        'description': info,
                        'args_schema': {}
                    })
                else:
                    # Bound method or other - FIXED: don't use str() on it
                    normalized.append({
                        'name': name,
                        'description': f"Tool: {name}",
                        'args_schema': {}
                    })

        elif isinstance(tool_methods, list):
            for tool in tool_methods:
                if isinstance(tool, dict):
                    # Already in dict format
                    if 'name' in tool:
                        normalized.append(tool)
                    else:
                        # Has keys but no 'name' - skip or warn
                        continue
                elif isinstance(tool, str):
                    # Just a tool name
                    normalized.append({
                        'name': tool,
                        'description': f'Tool: {tool}',
                        'args_schema': {}
                    })
                else:
                    normalized.append({
                        'name': str(tool),
                        'description': 'No description',
                        'args_schema': {}
                    })
        else:
            # Single tool
            normalized.append({
                'name': str(tool_methods),
                'description': 'No description',
                'args_schema': {}
            })
        
        return normalized

    def build_react_prompt(self) -> str:
        """
        Build a ReAct-style system prompt with STRICT format requirements.

        FIXED: Tool descriptions include argument schemas
        FIXED: Examples use real EnterpriseBench tool names

        Returns:
            Formatted system prompt string
        """
        # Header
        prompt = """You are an AI assistant that helps users by using tools. You MUST follow the ReAct format exactly.

AVAILABLE TOOLS:

"""

        # Tool descriptions with argument schemas
        tools_to_show = self.tool_methods[:20]  # Limit to 20 tools to save context
        
        for i, tool in enumerate(tools_to_show):
            tool_name = tool.get('name', 'unknown')
            tool_desc = tool.get('description', 'No description')
            args_schema = tool.get('args_schema', {})

            prompt += f"{i+1}. {tool_name}\n"
            prompt += f"   Description: {tool_desc}\n"
            
            # Add argument information if available
            if args_schema:
                prompt += f"   Arguments:\n"
                for arg_name, arg_info in args_schema.items():
                    arg_type = arg_info.get('type', 'string')
                    arg_desc = arg_info.get('description', '')
                    required = arg_info.get('required', False)
                    req_str = " (required)" if required else " (optional)"
                    prompt += f"     - {arg_name}: {arg_type}{req_str} - {arg_desc}\n"
            
            prompt += "\n"

        if len(self.tool_methods) > 20:
            prompt += f"... and {len(self.tool_methods) - 20} more tools available.\n\n"

        # Strict format instructions
        prompt += """
STRICT FORMAT REQUIREMENTS:

You MUST use this EXACT format for every response:

Thought: [Your reasoning about what to do next]
Action: [The tool name to use]
Action Input: {"param1": "value1", "param2": "value2"}

After the tool executes, you will see:

Observation: [Tool output]

Then continue with another Thought/Action cycle, OR provide final answer:

Final Answer: [Your final response to the user]

CRITICAL RULES:
1. ALWAYS start with "Thought:" (capital T, colon, space)
2. Action name must EXACTLY match a tool name from the list above
3. Action Input MUST be valid JSON with double quotes
4. Do NOT use XML tags like <think>, <action>, etc.
5. Do NOT add extra formatting or markdown
6. Each line starts with the keyword (Thought:, Action:, etc.)

EXAMPLE 1 - Reading an email:

User: Read email with ID email_123

Thought: I need to retrieve the email content using the read_email tool.
Action: read_email
Action Input: {"email_id": "email_123"}
Observation: {"subject": "Meeting Update", "body": "The meeting is at 3pm", "sender": "alice@company.com"}

Thought: I have retrieved the email content successfully.
Final Answer: The email with ID email_123 has subject "Meeting Update" from alice@company.com. The body says: "The meeting is at 3pm"

EXAMPLE 2 - Multi-step GitHub task:

User: List all issues in the repository called "myproject"

Thought: I need to get the list of issues for the repository named "myproject".
Action: github_list_issues_of_repository
Action Input: {"repo_name": "myproject"}
Observation: [{"id": "issue_1", "title": "Bug in login", "status": "Open"}, {"id": "issue_2", "title": "Add feature", "status": "Closed"}]

Thought: I have retrieved the list of issues. There are 2 issues total.
Final Answer: The repository "myproject" has 2 issues:
1. Issue issue_1: "Bug in login" (Open)
2. Issue issue_2: "Add feature" (Closed)

EXAMPLE 3 - Creating and updating a record:

User: Create a product called "Widget Pro" with ID prod_123, then update its price to $29.99

Thought: First I need to create the product with the given details.
Action: create_product
Action Input: {"product_id": "prod_123", "product_name": "Widget Pro"}
Observation: Product created successfully with ID prod_123

Thought: Now I need to update the product's price to $29.99.
Action: update_product
Action Input: {"product_id": "prod_123", "actual_price": "29.99"}
Observation: Product prod_123 updated successfully

Thought: Both steps completed successfully.
Final Answer: I created the product "Widget Pro" with ID prod_123 and updated its price to $29.99.

Now, follow this format exactly for all tasks!"""

        return prompt

    def build_compact_prompt(self) -> str:
        """
        Compact version for limited context scenarios.
        
        Returns:
            Compact prompt string
        """
        prompt = "Tools: "
        tool_names = [t.get('name', 'unknown') for t in self.tool_methods]
        prompt += ", ".join(tool_names[:30])
        
        if len(tool_names) > 30:
            prompt += f" (and {len(tool_names) - 30} more)"
        
        prompt += "\n\nFormat:\nThought: [reasoning]\nAction: [tool_name]\nAction Input: {\"arg\": \"value\"}\nObservation: [result]\nFinal Answer: [response]"
        
        return prompt

    def build_tool_list_only(self) -> str:
        """
        Build just the tool list (useful for custom prompts).

        Returns:
            Tool list string
        """
        tools_text = ""
        for tool in self.tool_methods:
            tool_name = tool.get('name', 'unknown')
            tool_desc = tool.get('description', '')
            args_schema = tool.get('args_schema', {})
            
            tools_text += f"- {tool_name}: {tool_desc}"
            
            if args_schema:
                args_list = []
                for arg_name, arg_info in args_schema.items():
                    req = "(required)" if arg_info.get('required', False) else "(optional)"
                    args_list.append(f"{arg_name} {req}")
                tools_text += f" | Args: {', '.join(args_list)}"
            
            tools_text += "\n"
        
        return tools_text

    def get_tool_count(self) -> int:
        """Get number of available tools."""
        return len(self.tool_methods)

    def get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [t.get('name', 'unknown') for t in self.tool_methods]
