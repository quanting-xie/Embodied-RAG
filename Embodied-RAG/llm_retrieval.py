import asyncio
from typing import List, Dict, Any
import networkx as nx
from llm import LLMInterface
from config import Config
import time
import logging
import json
from pathlib import Path

class ParallelLLMRetriever:
    def __init__(self, graph: nx.Graph, llm_interface: LLMInterface, max_parallel_paths: int = None):
        self.graph = graph
        self.llm = llm_interface
        self.max_parallel_paths = max_parallel_paths or Config.RETRIEVAL['max_parallel_paths']
        self.logger = logging.getLogger('retrieval')
        
        # Setup logging
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler('retrieval.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        print(f"Initialized ParallelLLMRetriever with {self.max_parallel_paths} parallel paths")

    async def retrieve_and_respond(self, query: str, query_type: str = "global") -> Dict:
        """Main method to retrieve context and generate response"""
        start_time = time.time()
        self.logger.info(f"\n=== Starting Retrieval for Query: '{query}' ===")
        
        try:
            # 1. Parallel Retrieval
            retrieved_nodes = await self.parallel_retrieve(query)
            retrieval_time = time.time() - start_time
            
            # 2. Build Context
            context = self._build_hierarchical_context(retrieved_nodes)
            
            # 3. Generate Response
            response = await self.generate_response(query, context, query_type)
            
            # 4. Extract Navigation Target (if applicable)
            target_position = self.extract_target_position(response)
            
            total_time = time.time() - start_time
            
            # Log completion
            self.logger.info(f"Total processing time: {total_time:.2f}s")
            self.logger.info(f"Retrieved {len(retrieved_nodes)} nodes")
            
            return {
                'response': response,
                'target_position': target_position,
                'retrieved_nodes': retrieved_nodes,
                'context': context,
                'timing': {
                    'retrieval': retrieval_time,
                    'total': total_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_and_respond: {str(e)}", exc_info=True)
            raise

    async def parallel_retrieve(self, query: str) -> List[str]:
        """Parallel retrieval of relevant nodes"""
        level_times = {}
        expanded_results = set()
        
        # Get nodes by level
        level_nodes = self._get_nodes_by_level()
        max_level = max(level_nodes.keys())
        
        # Initialize parallel paths
        paths = []
        for _ in range(self.max_parallel_paths):
            paths.append({
                'chain': [],
                'current_level': max_level
            })
        
        # Process levels in parallel
        async def process_path(path_info):
            chain = path_info['chain']
            current_level = path_info['current_level']
            
            while current_level >= 0:
                level_start = time.time()
                
                # Get available nodes
                available_nodes = self._get_available_nodes(
                    level_nodes[current_level], 
                    chain[-1] if chain else None
                )
                
                if not available_nodes:
                    break
                
                # LLM selection
                selected_node = await self._select_node(query, available_nodes)
                if not selected_node:
                    break
                
                chain.append(selected_node)
                level_times[current_level] = time.time() - level_start
                
                if self.graph.nodes[selected_node].get('type') == 'object':
                    break
                    
                current_level -= 1
            
            return chain
        
        # Process all paths in parallel
        path_results = await asyncio.gather(*[
            process_path(path) for path in paths
        ])
        
        # Collect results
        for chain in path_results:
            expanded_results.update(chain)
            
        # Log statistics
        self._log_retrieval_stats(level_times, expanded_results)
        
        return list(expanded_results)

    def _get_nodes_by_level(self) -> Dict[int, List[tuple]]:
        """Group nodes by level"""
        level_nodes = {}
        for node, data in self.graph.nodes(data=True):
            level = data.get('level', 0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append((node, data))
        return level_nodes

    def _get_available_nodes(self, level_nodes: List[tuple], previous_node: str) -> List[tuple]:
        """Get available nodes for selection"""
        if not previous_node:
            return level_nodes
            
        return [
            (n, d) for n, d in level_nodes
            if self.graph.has_edge(previous_node, n) and
            self.graph.edges[previous_node, n].get('relationship') == 'part_of'
        ]

    async def _select_node(self, query: str, nodes: List[tuple]) -> str:
        """LLM-based node selection"""
        nodes_for_selection = [
            {
                'id': node,
                'summary': data.get('summary', 'No summary'),
                'level': data.get('level', 0),
                'type': data.get('type', 'unknown'),
                'name': data.get('name', node)
            }
            for node, data in nodes
        ]
        
        context = await self.llm.generate_hierarchical_context(nodes_for_selection)
        return await self.llm.select_best_node(query, nodes_for_selection, context)

    def _build_hierarchical_context(self, nodes: List[str]) -> str:
        """Build hierarchical context from nodes"""
        if not nodes:
            return "No relevant nodes found."
            
        context = ["=== Hierarchical Structure ==="]
        
        # Group by level
        level_nodes = {}
        for node in nodes:
            data = self.graph.nodes[node]
            level = data.get('level', 0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append((node, data))
        
        # Build hierarchy
        for level in sorted(level_nodes.keys(), reverse=True):
            context.append(f"\nLevel {level}:")
            for node, data in level_nodes[level]:
                indent = "  " * (3 - level)
                context.append(f"{indent}{data.get('name', node)}")
                if 'summary' in data:
                    context.append(f"{indent}Summary: {data['summary']}")
        
        # Add object details
        context.append("\n=== Object Details ===")
        for node in nodes:
            data = self.graph.nodes[node]
            if data.get('type') == 'object':
                context.append(f"\nObject: {data.get('name', node)}")
                if 'position' in data:
                    context.append(f"Position: {self._format_position(data['position'])}")
                props = {k: v for k, v in data.items() 
                        if k not in ['embedding', 'position', 'label']}
                if props:
                    context.append(f"Properties: {json.dumps(props, indent=2)}")
        
        return "\n".join(context)

    async def generate_response(self, query: str, context: str, query_type: str) -> str:
        """Generate response using retrieved context"""
        try:
            return await self.llm.generate_navigation_response(query, context, query_type)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def extract_target_position(self, response: str) -> Dict:
        """Extract target position from response"""
        import re
        target_match = re.search(r'<<(.+?)>>', response)
        if target_match:
            target = target_match.group(1)
            for node, data in self.graph.nodes(data=True):
                if (data.get('name') == target or node == target) and 'position' in data:
                    return data['position']
        return None

    def _format_position(self, pos: Any) -> str:
        """Format position data consistently"""
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            return f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        elif isinstance(pos, dict) and all(k in pos for k in ['x', 'y', 'z']):
            return f"[{pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}]"
        return str(pos)

    def _log_retrieval_stats(self, level_times: Dict[int, float], results: set):
        """Log detailed retrieval statistics"""
        self.logger.info("\n=== Retrieval Statistics ===")
        total_time = sum(level_times.values())
        for level, time_taken in level_times.items():
            self.logger.info(
                f"Level {level}: {time_taken:.2f}s ({(time_taken/total_time)*100:.1f}%)"
            )
        self.logger.info(f"Retrieved Nodes: {len(results)}")
        for node in results:
            self.logger.info(f"- {node}") 