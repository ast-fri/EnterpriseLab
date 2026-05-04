# """
# Setup environment based on configuration
# Only loads the environments specified in config
# """
# import asyncio
# import os
# from dotenv import load_dotenv

# from environments.connector import UniversalConnector
# from environments.crm_arena import CRMArenaAdapter, CRMArenaLocalAdapter
# from environments import (
#     CRMArenaAdapter,
#     TauBenchRetailAdapter,
#     TauBenchAirlineAdapter,
#     TravelPlannerAdapter,
#     EnterpriseBenchAdapter,
#     EnterpriseArenaAdapter
# )


# from utils import GPTCaller


# async def main():
#     load_dotenv()
    
#     # Initialize GPT caller
#     gpt_caller = GPTCaller(
#         api_key=os.getenv("AZURE_CHAT_API_KEY"),
#         api_base=os.getenv("AZURE_CHAT_ENDPOINT"),
#         model_name="gpt-4o"
#     )
    
#     # Create universal connector
#     connector = UniversalConnector(gpt_caller=gpt_caller)
    
#     # Add CRMArena (live Salesforce API)
#     print("\n" + "="*70)
#     print("🏭 Adding CRMArena Original...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=CRMArenaAdapter,
#         config={
#             "org_type": "original",  # or "b2b", "b2c"
#             "crm_arena_path": "/EnterprisePlatform/TaskGenerationPipeline/environments/CRMArena",
#             "env_vars_loaded": True
#         }
#     )
#     # Add CRMArena Pro B2B
#     print("\n" + "="*70)
#     print("🏭 Adding CRMArena Pro B2B...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=CRMArenaAdapter,
#         config={
#             "org_type": "b2b",
#             "crm_arena_path": "/EnterprisePlatform/TaskGenerationPipeline/environments/CRMArena",
#             "env_vars_loaded": True
#         }
#     )
#     # Add CRMArena Pro B2C
#     print("\n" + "="*70)
#     print("🏭 Adding CRMArena Pro B2C...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=CRMArenaAdapter,
#         config={
#             "org_type": "b2c",
#             "crm_arena_path": "/EnterprisePlatform/TaskGenerationPipeline/environments/CRMArena",
#             "env_vars_loaded": True
#         }
#     )
#     # Add Tau-bench Retail
#     print("\n" + "="*70)
#     print("🛒 Adding Tau-bench Retail...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=TauBenchRetailAdapter,
#         config={
#             "tau_bench_path": "./environments/tau-bench"
#         }
#     )
    
#     # Add Tau-bench Airline
#     print("\n" + "="*70)
#     print("✈️  Adding Tau-bench Airline...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=TauBenchAirlineAdapter,
#         config={
#             "tau_bench_path": "./environments/tau-bench"
#         }
#     )
#     # Add TravelPlanner
#     print("\n" + "="*70)
#     print("🗺️  Adding TravelPlanner...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=TravelPlannerAdapter,
#         config={
#             "travel_planner_path": "./environments/TravelPlanner"
#         }
#     )
    
#     # Add EnterpriseBench
#     print("\n" + "="*70)
#     print("🏭 Adding EnterpriseBench...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=EnterpriseBenchAdapter,
#         config={
#             "enterprise_bench_path": "./environments/EnterpriseBench"
#         }
#     )
    
#     # 6. Add EnterpriseArena (MCP servers)
#     print("\n" + "="*70)
#     print("🚀 Adding EnterpriseArena (MCP)...")
#     print("="*70)
#     connector.add_environment(
#         adapter_class=EnterpriseArenaAdapter,
#         config={
#             "mcp_config_path": "./mcp_config_http.json",
#             "connection_timeout": 15.0,
#             "tool_load_timeout": 10.0
#         }
#     )
#     # Get summary
#     print("\n" + "="*70)
#     summary = connector.get_summary()
#     print(f"📊 Summary:")
#     print(f"   Total Environments: {summary['total_environments']}")
#     print(f"   Total Tools: {summary['total_tools']}")
#     print("\n   Environment Details:")
#     for env_name, details in summary['environments'].items():
#         print(f"   • {env_name}: {details['num_tools']} tools")
#     print("="*70 + "\n")
    
#     # Get all tools for AutoQuest
#     all_tools = connector.get_all_tools()
    
#     # Now run AutoQuest pipeline
#     # from edge_discovery import EdgeDiscovery
#     # from intelligent_explorer import IntelligentExplorer
    
#     # # Phase 1: Edge Discovery
#     # print("🔍 Starting Phase 1: Edge Discovery...")
#     # edge_discovery = EdgeDiscovery(
#     #     tools=all_tools,
#     #     gpt_caller=gpt_caller
#     # )
#     # initial_graph = await edge_discovery.build_graph()
    
#     # # Phase 2: Intelligent Exploration
#     # print("🧭 Starting Phase 2: Intelligent Exploration...")
#     # explorer = IntelligentExplorer(
#     #     tools=all_tools,
#     #     initial_graph=initial_graph,
#     #     gpt_caller=gpt_caller
#     # )
#     # trajectories, refined_graph = await explorer.explore()
    
#     # Cleanup
#     connector.cleanup()


# if __name__ == "__main__":
#     asyncio.run(main())



"""
Setup environment based on configuration
Only loads the environments specified in config
"""
import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from environments.connector import UniversalConnector
from environments import (
    CRMArenaAdapter,
    CRMArenaLocalAdapter,
    TauBenchRetailAdapter,
    TauBenchAirlineAdapter,
    TravelPlannerAdapter,
    EnterpriseBenchAdapter,
    EnterpriseArenaAdapter
)
from utils import GPTCaller


class EnvironmentSetup:
    """
    Setup environments based on configuration
    Dynamically loads only requested environments
    """
    
    # Environment registry
    ENVIRONMENT_REGISTRY = {
        "crm_arena": {
            "adapter": CRMArenaAdapter,
            "default_config": {
                "org_type": "original",
                "crm_arena_path": "./environments/CRMArena",
                "env_vars_loaded": True
            }
        },
       
        "tau_bench_retail": {
            "adapter": TauBenchRetailAdapter,
            "default_config": {
                "tau_bench_path": "./environments/tau-bench"
            }
        },
        "tau_bench_airline": {
            "adapter": TauBenchAirlineAdapter,
            "default_config": {
                "tau_bench_path": "./environments/tau-bench"
            }
        },
        "travel_planner": {
            "adapter": TravelPlannerAdapter,
            "default_config": {
                "travel_planner_path": "./environments/TravelPlanner",
                "load_planner": False
            }
        },
        "enterprise_bench": {
            "adapter": EnterpriseBenchAdapter,
            "default_config": {
                "enterprise_bench_path": "./environments/EnterpriseBench"
            }
        },
        "enterprise_arena": {
            "adapter": EnterpriseArenaAdapter,
            "default_config": {
                "mcp_config_path": "./mcp_config_http.json",
                "connection_timeout": 15.0,
                "tool_load_timeout": 10.0
            }
        }
    }
    
    def __init__(self, gpt_caller: GPTCaller):
        """
        Args:
            gpt_caller: GPT caller instance
        """
        self.gpt_caller = gpt_caller
        self.connector = UniversalConnector(gpt_caller=gpt_caller)
    
    def setup_environments(self, environment_configs: List[Dict[str, Any]]):
        """
        Setup environments based on configuration
        
        Args:
            environment_configs: List of environment configs
                Each config: {
                    "name": "environment_name",
                    "config": {...}  # Optional: override default config
                }
        """
        print("\n🌍 Setting up environments...")
        print("="*70)
        
        for env_config in environment_configs:
            env_name = env_config["name"]
            
            if env_name not in self.ENVIRONMENT_REGISTRY:
                print(f"⚠️  Unknown environment: {env_name}")
                continue
            
            # Get environment info from registry
            env_info = self.ENVIRONMENT_REGISTRY[env_name]
            adapter_class = env_info["adapter"]
            default_config = env_info["default_config"]
            
            # Merge with user config
            final_config = {**default_config, **env_config.get("config", {})}
            
            # Add environment
            print(f"\n🔧 Adding {env_name}...")
            success = self.connector.add_environment(
                adapter_class=adapter_class,
                config=final_config
            )
            
            if success:
                print(f"✅ {env_name} loaded successfully")
            else:
                print(f"❌ Failed to load {env_name}")
        
        print("\n" + "="*70)
        
        # Print summary
        summary = self.connector.get_summary()
        print(f"\n📊 Environment Setup Summary:")
        print(f"   • Total environments: {summary['total_environments']}")
        print(f"   • Total tools: {summary['total_tools']}")
        print(f"\n   Environment Details:")
        for env_name, details in summary['environments'].items():
            print(f"   • {env_name}: {details['num_tools']} tools")
        print("\n" + "="*70 + "\n")
        
        return self.connector
    
    def get_connector(self) -> UniversalConnector:
        """Get the universal connector"""
        return self.connector


async def setup_from_config(config: Dict[str, Any]) -> UniversalConnector:
    """
    Setup environments from configuration dict
    
    Args:
        config: Configuration with 'environments' key
    
    Returns:
        UniversalConnector with loaded environments
    """
    load_dotenv()
    
    # Initialize GPT caller
    gpt_caller = GPTCaller(
        api_key=os.getenv("AZURE_CHAT_API_KEY"),
        api_base=os.getenv("AZURE_CHAT_ENDPOINT"),
        model_name=config.get("model_name", "gpt-4o")
    )
    
    # Setup environments
    setup = EnvironmentSetup(gpt_caller)
    connector = setup.setup_environments(config.get("environments", []))
    
    return connector, gpt_caller
