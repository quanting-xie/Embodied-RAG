from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import asyncio
from retrieval_and_generation import EnvironmentalChat
import json
from tqdm import tqdm
from datetime import datetime
import googlemaps
import os
import scipy.stats
import folium
from folium import plugins
import webbrowser
import uuid
import base64
import argparse
import traceback

class RetrievalEvaluator:
    def __init__(self, graph_path: str = None, vector_db_path: str = None, image_dir: str = None):
        self.chat = None
        self.query_locations_file = Path("query_locations.json")
        self.results_dir = Path(__file__).parent.absolute() / "evaluation_results"
        self.results_dir.mkdir(exist_ok=True)
        self.k = 5  # Top-k results to compare
        self.radius = 5000  # Default search radius in meters
        
        # Set paths from arguments or use defaults, ensuring absolute paths
        self.graph_path = Path(graph_path).resolve() 
        self.vector_db_path = Path(vector_db_path).resolve()
        self.image_dir = Path(image_dir).resolve() 
        
        print(f"Using graph: {self.graph_path}")
        print(f"Using vector database: {self.vector_db_path}")
        print(f"Using images from: {self.image_dir}")
        
        # Validate paths
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        
        # Create directories if they don't exist
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Default center location (Pittsburgh)
        self.default_center = {
            'latitude': 40.4433,
            'longitude': -79.9436
        }
        
        # Enhanced vector_db path validation
        if self.vector_db_path.exists():
            files = list(self.vector_db_path.glob('**/*'))  # Recursively list all files
            print("\nVector Database Directory Contents:")
            print(f"Path: {self.vector_db_path}")
            print(f"Total files found: {len(files)}")
            for f in files:
                print(f"  - {f.relative_to(self.vector_db_path)}")
                if f.is_file():
                    print(f"    Size: {f.stat().st_size:,} bytes")
    
    async def initialize(self):
        """Initialize the EnvironmentalChat system with specific graph and vector db"""
        # Update config for paths
        import config
        config.Config.PATHS = {
            'graph_path': str(self.graph_path.resolve()),  # Ensure absolute path
            'vector_db_path': str(self.vector_db_path.resolve()),
            'image_dir': str(self.image_dir.resolve())
            # Remove forest_dir as it might be causing confusion
        }
        
        print(f"\nInitializing with absolute paths:")
        for key, path in config.Config.PATHS.items():
            print(f"{key}: {path}")
        
        # Create EnvironmentalChat instance
        self.chat = await EnvironmentalChat.create()
        
    async def interactive_mode(self):
        """Interactive mode for single query evaluation with multiple input styles"""
        print("\n=== Interactive Query Mode ===")
        print("\nInput styles:")
        print("1. Query only (e.g., 'Where are the convenience stores?')")
        print("2. Query + Location (use format 'L: lat,lon | Q: your question')")
        print("3. Query + Location + History (use format 'L: lat,lon | H: true | Q: your question')")
        print("\nCommands:")
        print("- 'quit': Exit the chat")
        print("- 'clear': Clear chat history")
        
        # Get user input
        user_input = input("\nEnter your query: ").strip()
        
        if user_input.lower() == 'quit':
            return False
        elif user_input.lower() == 'clear':
            self.chat_history = []
            print("Chat history cleared.")
            return True
        
        # Parse input format
        query = user_input
        location = self.default_center
        use_history = False
        
        if '|' in user_input:
            parts = [p.strip() for p in user_input.split('|')]
            for part in parts:
                if part.startswith('L:'):
                    try:
                        lat, lon = map(float, part[2:].strip().split(','))
                        location = {
                            'latitude': lat,
                            'longitude': lon
                        }
                    except:
                        print("Invalid location format. Using default location (Tokyo).")
                elif part.startswith('H:'):
                    use_history = part[2:].strip().lower() == 'true'
                elif part.startswith('Q:'):
                    query = part[2:].strip()
        
        # Create query item
        query_item = {
            'query': query,
            'location': location,
            'use_history': use_history,
            'searchable_term': query.split()[-1]  # Simple extraction of search term
        }
        
        # Evaluate query
        result = await self.evaluate_query(query_item)
        
        print("\nQuery Results:")
        print(json.dumps(result, indent=2))
        
        return True

    def visualize_results(self, query_item: Dict, retrieved_nodes: List[Dict], ground_truth: List[Dict]):
        """Visualize query results on an interactive map with clickable image popups"""
        # Create map centered on query location
        center_lat = query_item['location']['latitude']
        center_lon = query_item['location']['longitude']
        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)
        
        # Add marker for query location
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Query Location\n{query_item['query']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Add retrieved results with image popups
        for i, node in enumerate(retrieved_nodes):
            try:
                # Skip nodes without position data
                if not node.get('position'):
                    print(f"Skipping node {i} - no position data")
                    continue
                    
                # Get position data - handle both x/y and lat/lon formats
                position = node['position']
                if not isinstance(position, dict):
                    print(f"Skipping node {i} - invalid position format")
                    continue
                    
                # Try to get coordinates, skipping if not found
                try:
                    lat = float(position.get('y', position.get('latitude')))
                    lon = float(position.get('x', position.get('longitude')))
                except (TypeError, ValueError):
                    print(f"Skipping node {i} - invalid coordinate values")
                    continue
                    
                # Debug print node data
                print(f"\nProcessing node {i}:")
                print(json.dumps(node, indent=2))
                
                # Get node metadata
                name = node.get('name', 'Unnamed')
                caption = node.get('caption', 'No caption available')
                
                # Handle image path
                image_path = node.get('image_path', '')
                encoded_image = None
                
                if image_path:
                    # Use consistent path resolution
                    image_name = Path(image_path).name
                    absolute_image_path = self.image_dir / image_name  # Use self.image_dir directly
                    
                    print(f"Trying to load image from: {absolute_image_path}")
                    
                    # Try to read the image if path exists
                    if absolute_image_path.exists():
                        try:
                            with open(absolute_image_path, 'rb') as img_file:
                                encoded_image = base64.b64encode(img_file.read()).decode()
                        except Exception as e:
                            print(f"Error reading image file {absolute_image_path}: {str(e)}")
                    else:
                        print(f"Image not found at: {absolute_image_path}")
                
                # Create popup HTML
                popup_html = f"""
                <div style="width:800px; max-height:800px; overflow-y:auto;">
                    <h3 style="margin-bottom:10px;">Retrieved Result #{i+1}</h3>
                    <div style="margin-bottom:15px;">
                        <b>Name:</b> {name}<br>
                        <b>Score:</b> {node.get('score', 'N/A')}
                    </div>
                """
                
                if encoded_image:
                    popup_html += f"""
                    <div style="margin-bottom:15px;">
                        <img src="data:image/jpeg;base64,{encoded_image}" 
                             style="width:100%; max-width:800px; height:auto; margin-bottom:10px;">
                    </div>
                    """
                
                popup_html += f"""
                    <div style="margin-bottom:15px;">
                        <b>Caption:</b><br>
                        <div style="white-space:pre-wrap; padding:10px; background:#f5f5f5; border-radius:5px;">
                            {caption}
                        </div>
                    </div>
                    <div>
                        <b>Location:</b> {lat:.6f}, {lon:.6f}
                    </div>
                </div>
                """
                
                # Create marker with popup
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(folium.Html(popup_html, script=True), max_width=800),
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
                
            except Exception as e:
                print(f"Error processing node {i}:")
                print(f"Error: {str(e)}")
                print(f"Node data: {node}")
                continue
        
        # Add ground truth locations
        for i, place in enumerate(ground_truth):
            try:
                lat = place['position']['latitude']
                lon = place['position']['longitude']
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Ground Truth #{i+1}\n{place['name']}",
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)
            except Exception as e:
                print(f"Error adding ground truth marker {i}: {str(e)}")
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
             padding: 10px; border: 2px solid grey; border-radius: 5px">
        <p><i class="fa fa-map-marker fa-2x" style="color:red"></i> Query Location</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:blue"></i> Retrieved Results</p>
        <p><i class="fa fa-map-marker fa-2x" style="color:green"></i> Ground Truth</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save and open map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_file = self.results_dir / f"results_map_{timestamp}.html"
        m.save(str(map_file))
        webbrowser.open(f'file://{map_file}')

    def compute_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Compute haversine distance between two locations in meters"""
        R = 6371000  # Earth's radius in meters
        
        # Extract coordinates
        lat1 = loc1.get('y', loc1.get('latitude', 0))
        lon1 = loc1.get('x', loc1.get('longitude', 0))
        lat2 = loc2.get('latitude', 0)
        lon2 = loc2.get('longitude', 0)
        
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    def compute_haversine_distance(self, loc1: Dict, loc2: Dict) -> float:
        """
        Compute haversine distance between two locations in meters
        
        Args:
            loc1, loc2: Dictionaries containing latitude/longitude coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth's radius in meters
        
        # Extract coordinates
        lat1 = loc1.get('y', loc1.get('latitude', 0))
        lon1 = loc1.get('x', loc1.get('longitude', 0))
        lat2 = loc2.get('latitude', 0)
        lon2 = loc2.get('longitude', 0)
        
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

    async def get_ground_truth_locations(self, query_item: Dict) -> List[Dict]:
        """Get k closest ground truth locations from Google Maps"""
        agent_location = query_item['location']
        searchable_term = query_item['searchable_term']
        
        # Search Google Maps for nearest locations
        places_result = self.gmaps.places_nearby(
            location=(agent_location['latitude'], agent_location['longitude']),
            keyword=searchable_term.strip('"'),  # Remove quotes if present
            rank_by='distance'  # This will sort by distance automatically
        )
        
        # Extract k closest locations
        ground_truth_locations = []
        for place in places_result.get('results', [])[:self.k]:
            location = place['geometry']['location']
            ground_truth_locations.append({
                'name': place['name'],
                'position': {
                    'latitude': location['lat'],
                    'longitude': location['lng']
                },
                'place_id': place['place_id']
            })
            
        return ground_truth_locations

    async def compute_semantic_relativity(self, query: str, retrieved_nodes: List[Dict]) -> Dict[str, float]:
        """Compute semantic relativity using GPT-4 to score relevance"""
        system_prompt = """You are an expert evaluator. Rate the relevance of the location given a user's query on a scale of 1-5, where:
        1 = Completely irrelevant
        2 = Somewhat irrelevant
        3 = Moderately relevant
        4 = Very relevant
        5 = Perfectly relevant
        
        Consider both the visual content and the hierarchical context in your evaluation.
        Return only the numerical score without explanation."""

        async def get_score_for_node(node, attempt_num):
            try:
                # Get image path
                image_path = node.get('image_path', '')
                if not image_path:
                    print(f"Attempt {attempt_num}: No image path available")
                    return None

                # Use consistent path resolution
                image_name = Path(image_path).name
                absolute_image_path = self.image_dir / image_name  # Use self.image_dir directly
                
                print(f"Looking for image at: {absolute_image_path}")

                if not absolute_image_path.exists():
                    print(f"Attempt {attempt_num}: Image not found at {absolute_image_path}")
                    return None

                # Read and encode image
                try:
                    with open(absolute_image_path, 'rb') as img_file:
                        encoded_image = base64.b64encode(img_file.read()).decode()
                except Exception as e:
                    print(f"Attempt {attempt_num}: Error reading image file: {str(e)}")
                    return None

                # Format location with image and hierarchical context
                location_text = (
                    f"Location Name: {node.get('name', 'Unnamed')}\n"
                    f"Parent Areas: {node.get('parent_areas', [])}\n"
                    f"Visual Content: [Image Attached]\n"
                    f"Description: {node.get('caption', 'No description')}"
                )

                prompt = f"""Query: {query}

Location Information:
{location_text}

Rate the relevance of this location on a scale of 1-5:"""

                # Get score from GPT-4 with image
                response = await self.chat.llm.generate_response(
                    prompt, 
                    system_prompt,
                    image_base64=encoded_image
                )
                try:
                    score = float(response.strip())
                    if 1 <= score <= 5:
                        print(f"Attempt {attempt_num}: Score = {score}")
                        return score
                    else:
                        print(f"Attempt {attempt_num}: Invalid score range: {score}")
                except ValueError:
                    print(f"Attempt {attempt_num}: Invalid response format: {response}")
                
            except Exception as e:
                print(f"Attempt {attempt_num}: Error: {str(e)}")
                traceback.print_exc()
            return None

        scores_per_node = []
        for i, node in enumerate(retrieved_nodes[:5]):  # Only evaluate top 5
            node_scores = []
            num_attempts = 5
            
            for attempt in range(num_attempts):
                score = await get_score_for_node(node, attempt + 1)
                if score is not None:
                    node_scores.append(score)
                    print(f"Node {i+1}, Attempt {attempt + 1}: Score = {score}")
            
            if node_scores:
                # Average the scores for this node
                avg_score = sum(node_scores) / len(node_scores)
                scores_per_node.append(avg_score)
                print(f"Node {i+1} Average Score: {avg_score:.4f}")

        if not scores_per_node:
            return {'top1': 0.0, 'top5': 0.0}

        # Normalize scores to 0-1 range
        normalized_scores = [float(score - 1) / 4 for score in scores_per_node]
        
        return {
            'top1': normalized_scores[0] if normalized_scores else 0.0,
            'top5': sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
        }

    def compute_spatial_relativity(self, query_location: Dict, retrieved_nodes: List[Dict]) -> float:
        """
        Compute spatial relativity using exponential decay without a hard threshold
        """
        if not retrieved_nodes:
            return 0.0

        distances = []
        decay_factor = 0.005  # Controls how quickly the score decays with distance
                         # Smaller value = slower decay, larger value = faster decay

        for node in retrieved_nodes[:self.k]:
            try:
                # Get node location
                node_loc = {
                    'latitude': node['position'].get('y', node['position'].get('latitude')),
                    'longitude': node['position'].get('x', node['position'].get('longitude'))
                }
                
                # Compute distance
                distance = self.compute_haversine_distance(query_location, node_loc)
                
                # Use exponential decay: score = e^(-decay_factor * distance)
                score = np.exp(-decay_factor * distance)
                distances.append(score)
                
                # Debug output
                print(f"Distance: {distance:.1f}m, Score: {score:.4f}")
                
            except Exception as e:
                print(f"Error computing distance: {str(e)}")
                distances.append(0.0)

        return sum(distances) / len(distances) if distances else 0.0

    async def evaluate_generated_response(self, query: str, generated_response: str, retrieved_nodes: List[Dict]) -> Dict:
        try:
            # Parse the generated response as JSON
            try:
                response_data = json.loads(generated_response.strip('`json\n'))  # Remove markdown formatting
                generated_node = {
                    'name': response_data.get('name', ''),
                    'caption': response_data.get('caption', ''),
                    'position': response_data.get('position', {}),
                    'image_path': response_data.get('image_path', ''),
                    'parent_areas': response_data.get('parent_areas', []),
                    'reasons': response_data.get('reasons', '')
                }
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Warning: Error parsing generated response: {str(e)}")
                print(f"Generated response: {generated_response}")
                return None

            print("\n=== Evaluating Generated Response ===")
            print(f"Generated Location: {generated_node['name']}")
            
            # Score the generated result using the same image directory
            image_name = Path(generated_node['image_path']).name
            generated_node['image_path'] = str(self.image_dir / image_name)
            
            # Get semantic score with multiple attempts
            semantic_scores = []
            num_attempts = 5
            for attempt in range(num_attempts):
                score = await self.compute_semantic_relativity(query, [generated_node])
                if score['top1'] is not None:
                    semantic_scores.append(score['top1'])
                    print(f"Attempt {attempt + 1} Semantic Score: {score['top1']:.4f}")
            
            # Average the semantic scores
            avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
            
            # Get spatial score using retrieved nodes for comparison
            spatial_score = self.compute_spatial_relativity(
                {'latitude': float(generated_node['position']['y']), 
                 'longitude': float(generated_node['position']['x'])}, 
                retrieved_nodes  # Use retrieved nodes instead of [generated_node]
            )
            
            # Calculate combined score
            final_score = avg_semantic_score * spatial_score
            
            generation_evaluation = {
                'semantic_score': avg_semantic_score,
                'spatial_score': spatial_score,
                'combined_score': final_score
            }
            
            print("\n=== Generation Evaluation Results ===")
            print(f"Average Semantic Score ({len(semantic_scores)} attempts): {avg_semantic_score:.4f}")
            print(f"Spatial Score: {spatial_score:.4f}")
            print(f"Combined Score: {final_score:.4f}")
            
            return generation_evaluation
            
        except Exception as e:
            print(f"Error evaluating generated response: {str(e)}")
            traceback.print_exc()
            return None

    async def evaluate_query(self, query_item: Dict) -> Dict:
        """Modified evaluate_query to include generation evaluation"""
        query = query_item['query']
        agent_location = query_item['location']
        use_history = query_item.get('use_history', False)
        
        try:
            print("\nRetrieving context...")
            # Get both the nodes and the hierarchical context
            context = await self.chat.retrieve_hierarchical_context(
                query=query,
                agent_location=agent_location,
                return_nodes=False  # Get the formatted hierarchical context
            )
            
            # Get nodes separately for evaluation
            retrieved_nodes = await self.chat.retrieve_hierarchical_context(
                query=query,
                agent_location=agent_location,
                return_nodes=True
            )
            
            print(f"\nFound {len(retrieved_nodes)} locations")
            if retrieved_nodes:
                print("\nFirst retrieved location:")
                print(json.dumps(retrieved_nodes[0], indent=2))
            
            # Generate response using the hierarchical context
            if use_history:
                response = await self.chat.generate_response(query, context)
            else:
                response = await self.chat.generate_response_no_history(query, context)
            
            # Compute retrieval metrics
            semantic_scores = await self.compute_semantic_relativity(query, retrieved_nodes)
            spatial_score = self.compute_spatial_relativity(agent_location, retrieved_nodes)
            
            # Evaluate generated response
            generation_scores = await self.evaluate_generated_response(query, response, retrieved_nodes)
            
            # Calculate final semantic-spatial scores
            final_scores = {
                'top1': semantic_scores['top1'] * spatial_score,
                'top5': semantic_scores['top5'] * spatial_score
            }
            
            # Visualize results
            if retrieved_nodes:
                self.visualize_results(query_item, retrieved_nodes[:self.k], [])
            
            return {
                'query': query,
                'agent_location': agent_location,
                'use_history': use_history,
                'response': response,
                'retrieved_nodes': retrieved_nodes[:self.k],
                'metrics': {
                    'semantic_relativity': semantic_scores,
                    'spatial_relativity': spatial_score,
                    'semantic_spatial_score': final_scores,
                    'generation_evaluation': generation_scores
                },
                'retrieved_count': len(retrieved_nodes),
                'success': len(retrieved_nodes) > 0
            }
            
        except Exception as e:
            print(f"Error evaluating query: {str(e)}")
            print(traceback.format_exc())
            return {
                'error': str(e),
                'success': False
            }

    async def run_evaluation(self):
        """Run evaluation on all queries and compute aggregate metrics"""
        results = []
        query_items = self.query_data['query_locations']
        
        print(f"\nEvaluating {len(query_items)} queries...")
        for query_item in tqdm(query_items):
            result = await self.evaluate_query(query_item)
            results.append(result)
        
        # Compute overall metrics with confidence intervals
        def compute_ci(values, confidence=0.95):
            n = len(values)
            mean = np.mean(values)
            se = np.std(values, ddof=1) / np.sqrt(n)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2, n-1)
            return mean, h
        
        overall_metrics = {
            'total_queries': len(results),
            'successful_queries': len([r for r in results if r['success']]),
            'metrics': {
                'semantic_relativity': {
                    'top1': compute_ci([r['metrics']['semantic_relativity']['top1'] for r in results]),
                    'top5': compute_ci([r['metrics']['semantic_relativity']['top5'] for r in results])
                },
                'spatial_relativity': compute_ci([r['metrics']['spatial_relativity'] for r in results]),
                'semantic_spatial_score': {
                    'top1': compute_ci([r['metrics']['semantic_spatial_score']['top1'] for r in results]),
                    'top5': compute_ci([r['metrics']['semantic_spatial_score']['top5'] for r in results])
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return overall_metrics

async def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the evaluation system')
    parser.add_argument('--graph-dir', type=str, help='Path to the semantic forest graph file')
    parser.add_argument('--vector-db', type=str, help='Path to the vector database directory')
    parser.add_argument('--image-dir', type=str, help='Path to the images directory')
    parser.add_argument('--mode', type=str, choices=['batch', 'interactive'], 
                       default='interactive', help='Evaluation mode')
    
    args = parser.parse_args()
    
    # Initialize evaluator with provided paths
    evaluator = RetrievalEvaluator(
        graph_path=args.graph_dir,
        vector_db_path=args.vector_db,
        image_dir=args.image_dir
    )
    await evaluator.initialize()
    
    if args.mode == 'batch':
        print("\nStarting batch evaluation...")
        results = await evaluator.run_evaluation()
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))
        print("\nDetailed results saved to evaluation_results directory")
    
    else:  # interactive mode
        while True:
            await evaluator.interactive_mode()
            if input("\nTry another query? [Y/n]: ").lower() == 'n':
                break

if __name__ == "__main__":
    asyncio.run(main()) 