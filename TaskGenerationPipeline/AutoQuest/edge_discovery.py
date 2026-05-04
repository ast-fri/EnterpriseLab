"""
Phase 1: Edge Discovery - Build initial tool dependency graph using LLM
WITH DOMAIN-BASED EDGE CONSTRAINTS
"""

import json
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ToolNode:
    """Represents a tool node in the dependency graph"""
    name: str
    description: str
    args_schema: Dict
    return_schema: Dict
    inferred_outputs: Dict
    required_inputs: Dict
    domain: str = "unknown"  # Added domain field
    out_degree: int = 0
    in_degree: int = 0
    centrality: int = 0
    execution_count: int = 0
    successful_executions: List[Dict] = None
    
    def __post_init__(self):
        if self.successful_executions is None:
            self.successful_executions = []


@dataclass
class ToolEdge:
    """Represents a dependency edge between tools"""
    source: str
    target: str
    confidence: float
    field_mappings: List[Dict]
    workflow_scenario: str
    status: str = "inferred"  # inferred, verified, rejected
    successful_uses: int = 0
    failed_uses: int = 0
    discovered_during_exploration: bool = False
    cross_domain: bool = False  # Added flag for cross-domain edges


class EdgeDiscovery:
    """
    Phase 1: Discover tool dependencies using LLM-based inference
    WITH DOMAIN-BASED CONSTRAINTS
    """
    
    # Communication domains that can cross-connect
    COMMUNICATION_DOMAINS = {"email", "messaging", "notification", "communication"}
    
    def __init__(
        self,
        tools: List[Any],
        gpt_caller: callable,
        batch_size: int = 50,
        edge_batch_size: int = 100,
        confidence_threshold: float = 0.5,
        allow_cross_domain_communication: bool = True  # New parameter
    ):
        """
        Args:
            tools: List of MCP tools with name, description, arg_schema, server_name
            gpt_caller: Async function to call GPT (takes prompt, returns JSON)
            batch_size: Number of tools to analyze per batch
            edge_batch_size: Number of tool pairs to analyze per batch
            confidence_threshold: Minimum confidence to include an edge
            allow_cross_domain_communication: Allow edges between communication tools
        """
        self.tools = tools
        self.gpt_caller = gpt_caller
        self.batch_size = batch_size
        self.edge_batch_size = edge_batch_size
        self.confidence_threshold = confidence_threshold
        self.allow_cross_domain_communication = allow_cross_domain_communication
        
        self.graph = {
            "nodes": {},
            "edges": {},
            "metadata": {
                "creation_method": "llm_heuristic_domain_constrained",
                "confidence": "initial",
                "needs_refinement": True,
                "total_tools": len(tools),
                "domain_constraint_enabled": True,
                "allow_cross_domain_communication": allow_cross_domain_communication
            }
        }
        
        # Track domains
        self.domain_clusters = {}
    
    async def build_graph(self) -> Dict[str, Any]:
        """
        Main method to build the initial dependency graph
        
        Returns:
            Complete dependency graph with nodes and edges
        """
        print("🔨 Phase 1: Building Initial Dependency Graph (Domain-Constrained)")
        print("=" * 60)
        
        # Step 1: Infer tool schemas and semantics
        await self._infer_tool_schemas()
        
        # Step 2: Build nodes
        self._build_nodes()
        
        # Step 3: Cluster by domain
        self._cluster_by_domain()
        
        # Step 4: Infer edges (dependencies) with domain constraints
        await self._infer_edges_domain_constrained()
        
        # Step 5: Compute graph statistics
        self._compute_statistics()
        
        # Step 6: Identify key nodes
        self._identify_key_nodes()
        
        self._print_summary()
        
        return self.graph
    
    async def _infer_tool_schemas(self):
        """Infer output schemas and semantics for all tools using LLM"""
        print("\n📋 Step 1: Inferring tool schemas and semantics...")
        
        self.tool_profiles = []
        
        for i in range(0, len(self.tools), self.batch_size):
            batch = self.tools[i:i+self.batch_size]
            
            batch_analysis_prompt = self._create_tool_analysis_prompt(batch)
            
            try:
                batch_analysis = await self.gpt_caller(
                    prompt=batch_analysis_prompt,
                    response_format="json",
                    model="gpt-4",
                    temperature=0.3
                )
                
                self.tool_profiles.extend(batch_analysis["tools_analysis"])
                
                print(f"   ✓ Analyzed tools {i+1}-{min(i+self.batch_size, len(self.tools))}/{len(self.tools)}")
                
            except Exception as e:
                print(f"   ✗ Error analyzing batch {i}-{i+self.batch_size}: {e}")
                continue
    
    def _create_tool_analysis_prompt(self, batch: List[Any]) -> str:
        """Create prompt for analyzing a batch of tools"""
        
        tools_json = json.dumps([{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.args_schema,
            "return_schema": tool.return_schema
        } for tool in batch], indent=2)
        
        return f"""
You are analyzing tools in an agentic environment to build a domain-constrained dependency graph.

For each tool below, infer:
1. What data types it likely returns (object, array, string, number, boolean, null)
2. What fields/keys would be in the return value
3. **CRITICAL: What PRIMARY domain/category it belongs to** (The Domains are [HR, IT, Software Engineering, Sales(CRM), Business Management, Communication])
4. In order to check the domain/category of the tool, check it's name, description, input and return schema and classify in one of [HR, IT, Software Engineering, Sales(CRM), Business Management, Communication]
5. What entities it operates on (e.g., User, Email, File, Event, Project, Issue, Customer, Employee, etc.)
6. Common use cases for this tool

**IMPORTANT**: Each tool should have ONE primary domain. Be specific and consistent with domain naming.


Tools to analyze:
{tools_json}

Return JSON with this exact structure:
{{
  "tools_analysis": [
    {{
      "tool_name": "exact tool name",
      "domain": "primary_domain_name",
      "likely_returns": {{
        "type": "object|array|string|number|boolean|null",
        "fields": ["field1", "field2", "field3"],
        "example": {{"sample_field": "sample_value"}},
        "semantic_category": "category_name",
        "entities": ["Entity1", "Entity2"]
      }},
      "requires": {{
        "input_fields": ["input_field1", "input_field2"],
        "field_types": {{"field1": "string", "field2": "number"}},
        "entities": ["Entity1", "Entity2"]
      }},
      "use_cases": ["use case 1", "use case 2", "use case 3"]
    }}
  ]
}}
"""
    
    def _build_nodes(self):
        """Build graph nodes from tool profiles"""
        print("\n🔷 Step 2: Building graph nodes...")
        
        for profile in self.tool_profiles:
            tool = next((t for t in self.tools if t.name == profile["tool_name"]), None)
            
            if tool is None:
                print(f"   ⚠ Warning: Tool {profile['tool_name']} not found")
                continue
            
            node = ToolNode(
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
                return_schema=tool.return_schema,
                inferred_outputs=profile["likely_returns"],
                required_inputs=profile["requires"],
                domain=profile.get("domain", "unknown")
            )
            
            self.graph["nodes"][tool.name] = node
        
        print(f"   ✓ Created {len(self.graph['nodes'])} nodes")
    
    def _cluster_by_domain(self):
        """Cluster tools by domain"""
        print("\n🗂️  Step 3: Clustering tools by domain...")
        
        for node_name, node in self.graph["nodes"].items():
            domain = node.domain
            if domain not in self.domain_clusters:
                self.domain_clusters[domain] = []
            self.domain_clusters[domain].append(node_name)
        
        print(f"   ✓ Found {len(self.domain_clusters)} domains:")
        for domain, tools in sorted(self.domain_clusters.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"      • {domain}: {len(tools)} tools")
        
        self.graph["metadata"]["domain_clusters"] = {
            domain: len(tools) for domain, tools in self.domain_clusters.items()
        }
    
    def _can_connect_domains(self, source_domain: str, target_domain: str) -> bool:
        """
        Check if two domains can be connected
        
        Rules:
        1. Same domain -> always allowed
        2. Both communication domains -> allowed if flag enabled
        3. One communication, one non-communication -> allowed if common parameters
        4. Different non-communication domains -> NOT allowed
        """
        # Same domain always allowed
        if source_domain == target_domain:
            return True
        
        # Check if both are communication domains
        source_is_comm = source_domain.lower() in self.COMMUNICATION_DOMAINS
        target_is_comm = target_domain.lower() in self.COMMUNICATION_DOMAINS
        
        if source_is_comm and target_is_comm and self.allow_cross_domain_communication:
            return True
        
        # One is communication, one is not -> will check parameters later
        if source_is_comm or target_is_comm:
            return True  # Will validate with parameter check in edge inference
        
        # Both are non-communication and different -> NOT allowed
        return False
    
    async def _infer_edges_domain_constrained(self):
        """Infer edges (dependencies) between tools WITH DOMAIN CONSTRAINTS"""
        print("\n🔗 Step 4: Inferring tool dependencies (domain-constrained)...")
        
        # Create candidate pairs based on domain constraints
        tool_names = list(self.graph["nodes"].keys())
        potential_pairs = []
        
        for source in tool_names:
            for target in tool_names:
                if source == target:
                    continue
                
                source_domain = self.graph["nodes"][source].domain
                target_domain = self.graph["nodes"][target].domain
                
                # Apply domain constraint
                if self._can_connect_domains(source_domain, target_domain):
                    potential_pairs.append((source, target))
        
        print(f"   • Analyzing {len(potential_pairs)} potential dependencies (after domain filtering)...")
        print(f"   • Filtered out {len(tool_names) * (len(tool_names) - 1) - len(potential_pairs)} cross-domain pairs")
        
        edges_found = 0
        
        for i in range(0, len(potential_pairs), self.edge_batch_size):
            batch_pairs = potential_pairs[i:i+self.edge_batch_size]
            
            edge_inference_prompt = self._create_edge_analysis_prompt_domain_aware(batch_pairs)
            
            try:
                edge_analysis = await self.gpt_caller(
                    prompt=edge_inference_prompt,
                    response_format="json",
                    model="gpt-4",
                    temperature=0.3
                )
                
                # Add edges with sufficient confidence
                for dep in edge_analysis["dependencies"]:
                    if dep["has_dependency"] and dep["confidence"] > self.confidence_threshold:
                        source_domain = self.graph["nodes"][dep["source_tool"]].domain
                        target_domain = self.graph["nodes"][dep["target_tool"]].domain
                        
                        # Final validation for cross-domain communication edges
                        is_cross_domain = source_domain != target_domain
                        
                        if is_cross_domain:
                            # Require common parameters for cross-domain edges
                            if not dep.get("has_common_parameters", False):
                                continue
                        
                        edge_key = f"{dep['source_tool']}->{dep['target_tool']}"
                        
                        edge = ToolEdge(
                            source=dep["source_tool"],
                            target=dep["target_tool"],
                            confidence=dep["confidence"],
                            field_mappings=dep.get("field_mappings", []),
                            workflow_scenario=dep.get("workflow_scenario", ""),
                            status="inferred",
                            cross_domain=is_cross_domain
                        )
                        
                        self.graph["edges"][edge_key] = edge
                        edges_found += 1
                
                if (i + self.edge_batch_size) % 500 == 0:
                    print(f"   • Progress: {min(i+self.edge_batch_size, len(potential_pairs))}/{len(potential_pairs)} pairs analyzed, {edges_found} edges found")
                
            except Exception as e:
                print(f"   ✗ Error analyzing edge batch {i}-{i+self.edge_batch_size}: {e}")
                continue
        
        # Count cross-domain edges
        cross_domain_edges = sum(1 for edge in self.graph["edges"].values() if edge.cross_domain)
        
        print(f"   ✓ Found {len(self.graph['edges'])} dependencies")
        print(f"   ✓ Same-domain edges: {len(self.graph['edges']) - cross_domain_edges}")
        print(f"   ✓ Cross-domain edges (communication only): {cross_domain_edges}")
    
    def _create_edge_analysis_prompt_domain_aware(self, pairs: List[Tuple[str, str]]) -> str:
        """Create DOMAIN-AWARE prompt for analyzing tool dependency pairs"""
        
        pairs_data = []
        for source_name, target_name in pairs:
            source_profile = next((p for p in self.tool_profiles if p["tool_name"] == source_name), None)
            target_profile = next((p for p in self.tool_profiles if p["tool_name"] == target_name), None)
            
            if source_profile and target_profile:
                source_node = self.graph["nodes"][source_name]
                target_node = self.graph["nodes"][target_name]
                
                pairs_data.append({
                    "source": {
                        "name": source_name,
                        "domain": source_node.domain,
                        "outputs": source_profile["likely_returns"],
                        "use_cases": source_profile.get("use_cases", [])
                    },
                    "target": {
                        "name": target_name,
                        "domain": target_node.domain,
                        "inputs": target_profile["requires"],
                        "use_cases": target_profile.get("use_cases", [])
                    }
                })
        
        return f"""
You are building a DOMAIN-CONSTRAINED dependency graph for tools in an Agentic Environment.

**CRITICAL RULES:**
1. Tools from the SAME domain CAN have edges if their data flows make sense
2. Tools from DIFFERENT domains generally CANNOT have edges
3. EXCEPTION: Communication tools (email, messaging, notification) CAN connect to each other IF they share common parameters (e.g., recipient, subject, message body, attachment)
4. Do NOT create edges between completely different departments/domains (e.g., HR ↔ Finance, CRM ↔ Code Repository)

For each pair, analyze:
1. Are they in the same domain? If YES, evaluate dependency normally
2. Are they both communication tools? If YES, check for common parameters
3. Are they from different non-communication domains? If YES, return has_dependency: false

Tool pairs to analyze:
{json.dumps(pairs_data, indent=2)}

For each pair, return:
{{
  "dependencies": [
    {{
      "source_tool": "exact source tool name",
      "target_tool": "exact target tool name",
      "has_dependency": true/false,
      "confidence": 0.0-1.0,
      "reasoning": "brief explanation considering domain constraints",
      "field_mappings": [
        {{"source_field": "field_name", "target_field": "field_name", "match_type": "exact|semantic|type"}}
      ],
      "workflow_scenario": "1-2 sentence description of when this chain would be used",
      "has_common_parameters": true/false,
      "domain_compatibility": "same_domain|cross_domain_communication|incompatible"
    }}
  ]
}}

Only include has_dependency: true when:
- Same domain AND clear meaningful connection, OR
- Both communication domains AND common parameters exist

Reject cross-domain edges unless they're both communication tools with shared parameters.
"""
    
    def _compute_statistics(self):
        """Compute degree centrality and other statistics"""
        print("\n📊 Step 5: Computing graph statistics...")
        
        # Initialize degree counts
        out_degree = {name: 0 for name in self.graph["nodes"]}
        in_degree = {name: 0 for name in self.graph["nodes"]}
        
        # Count degrees
        for edge_key, edge in self.graph["edges"].items():
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1
        
        # Update nodes
        for node_name, node in self.graph["nodes"].items():
            node.out_degree = out_degree[node_name]
            node.in_degree = in_degree[node_name]
            node.centrality = out_degree[node_name] + in_degree[node_name]
        
        # Compute graph-level statistics
        self.graph["metadata"]["avg_out_degree"] = np.mean(list(out_degree.values()))
        self.graph["metadata"]["avg_in_degree"] = np.mean(list(in_degree.values()))
        
        # Compute per-domain statistics
        domain_stats = {}
        for domain, tool_names in self.domain_clusters.items():
            domain_edges = sum(
                1 for edge in self.graph["edges"].values()
                if self.graph["nodes"][edge.source].domain == domain
            )
            domain_stats[domain] = {
                "tool_count": len(tool_names),
                "edge_count": domain_edges,
                "avg_out_degree": domain_edges / len(tool_names) if tool_names else 0
            }
        
        self.graph["metadata"]["domain_statistics"] = domain_stats
        
        print(f"   ✓ Average out-degree: {self.graph['metadata']['avg_out_degree']:.2f}")
        print(f"   ✓ Average in-degree: {self.graph['metadata']['avg_in_degree']:.2f}")
    
    def _identify_key_nodes(self):
        """Identify most central/important nodes per domain"""
        print("\n🎯 Step 6: Identifying key nodes...")
        
        # Sort by centrality
        sorted_nodes = sorted(
            self.graph["nodes"].items(),
            key=lambda x: x[1].centrality,
            reverse=True
        )
        
        # Store top 20 most central nodes
        self.graph["metadata"]["most_central_nodes"] = [
            {
                "name": name,
                "domain": node.domain,
                "centrality": node.centrality,
                "out_degree": node.out_degree,
                "in_degree": node.in_degree,
            }
            for name, node in sorted_nodes[:20]
        ]
        
        print(f"   ✓ Top 5 most central tools:")
        for i, central_node in enumerate(self.graph["metadata"]["most_central_nodes"][:5], 1):
            print(f"      {i}. {central_node['name']} ({central_node['domain']}) - centrality: {central_node['centrality']}")
        
        # Find most central tool per domain
        domain_leaders = {}
        for domain, tool_names in self.domain_clusters.items():
            domain_tools = [(name, self.graph["nodes"][name]) for name in tool_names]
            if domain_tools:
                leader = max(domain_tools, key=lambda x: x[1].centrality)
                domain_leaders[domain] = {
                    "name": leader[0],
                    "centrality": leader[1].centrality
                }
        
        self.graph["metadata"]["domain_leaders"] = domain_leaders
    
    def _print_summary(self):
        """Print summary of graph construction"""
        print("\n" + "=" * 60)
        print("✅ Phase 1 Complete: Domain-Constrained Dependency Graph Built")
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   • Total tools (nodes): {len(self.graph['nodes'])}")
        print(f"   • Total domains: {len(self.domain_clusters)}")
        print(f"   • Total dependencies (edges): {len(self.graph['edges'])}")
        
        cross_domain_edges = sum(1 for edge in self.graph["edges"].values() if edge.cross_domain)
        print(f"   • Same-domain edges: {len(self.graph['edges']) - cross_domain_edges}")
        print(f"   • Cross-domain edges: {cross_domain_edges}")
        
        print(f"   • Average out-degree: {self.graph['metadata']['avg_out_degree']:.2f}")
        print(f"   • Average in-degree: {self.graph['metadata']['avg_in_degree']:.2f}")
        print(f"   • Edge confidence threshold: {self.confidence_threshold}")
        print(f"   • Domain constraint: ENABLED")
        print(f"   • Status: Ready for exploration and refinement")
        print("=" * 60 + "\n")
    
    def get_graph(self) -> Dict[str, Any]:
        """Return the constructed graph"""
        return self.graph
    
    def save_graph(self, filepath: str):
        """Save graph to JSON file"""
        import json
        
        # Convert dataclasses to dicts for JSON serialization
        serializable_graph = {
            "nodes": {
                name: {
                    "name": node.name,
                    "description": node.description,
                    "args_schema": node.args_schema,
                    "return_schema": node.return_schema,
                    "inferred_outputs": node.inferred_outputs,
                    "required_inputs": node.required_inputs,
                    "domain": node.domain,
                    "out_degree": node.out_degree,
                    "in_degree": node.in_degree,
                    "centrality": node.centrality
                }
                for name, node in self.graph["nodes"].items()
            },
            "edges": {
                key: {
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                    "field_mappings": edge.field_mappings,
                    "workflow_scenario": edge.workflow_scenario,
                    "status": edge.status,
                    "cross_domain": edge.cross_domain
                }
                for key, edge in self.graph["edges"].items()
            },
            "metadata": self.graph["metadata"],
            "domain_clusters": self.domain_clusters
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_graph, f, indent=2)
        
        print(f"💾 Graph saved to {filepath}")
