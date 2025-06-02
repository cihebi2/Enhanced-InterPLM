# 回路模式检测 
"""Motif detection and matching for circuit discovery."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path
import networkx as nx
from collections import defaultdict


@dataclass
class CircuitMotif:
    """Representation of a circuit motif pattern."""
    name: str
    pattern: nx.DiGraph
    biological_function: str
    conservation_score: float
    min_size: int
    max_size: int
    required_properties: Dict[str, any]


class MotifLibrary:
    """
    Library of known biological circuit motifs.
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        self.motifs = {}
        
        if library_path and library_path.exists():
            self.load_from_file(library_path)
        else:
            self._initialize_default_motifs()
            
    def _initialize_default_motifs(self):
        """Initialize with common biological motifs."""
        
        # Feed-forward motif
        ff_pattern = nx.DiGraph()
        ff_pattern.add_edges_from([(0, 1), (0, 2), (1, 2)])
        self.motifs['feed_forward'] = CircuitMotif(
            name='feed_forward',
            pattern=ff_pattern,
            biological_function='Signal amplification and filtering',
            conservation_score=0.85,
            min_size=3,
            max_size=5,
            required_properties={'coherent': True}
        )
        
        # Feedback loop
        fb_pattern = nx.DiGraph()
        fb_pattern.add_edges_from([(0, 1), (1, 2), (2, 0)])
        self.motifs['feedback_loop'] = CircuitMotif(
            name='feedback_loop',
            pattern=fb_pattern,
            biological_function='Homeostasis and oscillation',
            conservation_score=0.80,
            min_size=3,
            max_size=8,
            required_properties={'cyclic': True}
        )
        
        # Cascade motif
        cascade_pattern = nx.DiGraph()
        cascade_pattern.add_edges_from([(0, 1), (1, 2), (2, 3)])
        self.motifs['cascade'] = CircuitMotif(
            name='cascade',
            pattern=cascade_pattern,
            biological_function='Signal transduction',
            conservation_score=0.90,
            min_size=3,
            max_size=10,
            required_properties={'linear': True}
        )
        
        # Hub motif
        hub_pattern = nx.DiGraph()
        for i in range(1, 5):
            hub_pattern.add_edge(0, i)
        self.motifs['hub'] = CircuitMotif(
            name='hub',
            pattern=hub_pattern,
            biological_function='Central regulation',
            conservation_score=0.75,
            min_size=4,
            max_size=20,
            required_properties={'central_node': True}
        )
        
        # Bi-fan motif
        bifan_pattern = nx.DiGraph()
        bifan_pattern.add_edges_from([(0, 2), (0, 3), (1, 2), (1, 3)])
        self.motifs['bi_fan'] = CircuitMotif(
            name='bi_fan',
            pattern=bifan_pattern,
            biological_function='Coordinated regulation',
            conservation_score=0.70,
            min_size=4,
            max_size=6,
            required_properties={'bipartite': True}
        )
        
    def load_from_file(self, path: Path):
        """Load motif library from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        for motif_data in data['motifs']:
            pattern = nx.node_link_graph(motif_data['pattern'])
            motif = CircuitMotif(
                name=motif_data['name'],
                pattern=pattern,
                biological_function=motif_data['biological_function'],
                conservation_score=motif_data['conservation_score'],
                min_size=motif_data['min_size'],
                max_size=motif_data['max_size'],
                required_properties=motif_data['required_properties']
            )
            self.motifs[motif.name] = motif
            
    def save_to_file(self, path: Path):
        """Save motif library to JSON file."""
        data = {'motifs': []}
        
        for motif in self.motifs.values():
            motif_data = {
                'name': motif.name,
                'pattern': nx.node_link_data(motif.pattern),
                'biological_function': motif.biological_function,
                'conservation_score': motif.conservation_score,
                'min_size': motif.min_size,
                'max_size': motif.max_size,
                'required_properties': motif.required_properties
            }
            data['motifs'].append(motif_data)
            
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_motif(self, name: str) -> Optional[CircuitMotif]:
        """Get motif by name."""
        return self.motifs.get(name)
        
    def list_motifs(self) -> List[str]:
        """List all available motif names."""
        return list(self.motifs.keys())


class MotifMatcher:
    """
    Matches discovered circuits against known motif patterns.
    """
    
    def __init__(self, motif_library: MotifLibrary):
        self.library = motif_library
        
    def match_circuit(
        self,
        circuit_graph: nx.DiGraph,
        tolerance: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Match a circuit against all motifs in library.
        
        Args:
            circuit_graph: Circuit to match
            tolerance: Matching tolerance (0-1)
            
        Returns:
            List of (motif_name, match_score) tuples
        """
        matches = []
        
        for motif_name, motif in self.library.motifs.items():
            if self._check_size_compatibility(circuit_graph, motif):
                match_score = self._compute_match_score(circuit_graph, motif.pattern)
                
                if match_score >= tolerance:
                    matches.append((motif_name, match_score))
                    
        # Sort by match score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
        
    def find_motif_instances(
        self,
        graph: nx.DiGraph,
        motif_name: str,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Find all instances of a specific motif in a graph.
        
        Args:
            graph: Graph to search in
            motif_name: Name of motif to find
            min_score: Minimum match score
            
        Returns:
            List of motif instances with their properties
        """
        motif = self.library.get_motif(motif_name)
        if not motif:
            raise ValueError(f"Unknown motif: {motif_name}")
            
        instances = []
        
        # Get all subgraphs of appropriate size
        for nodes in self._get_subgraphs(graph, len(motif.pattern)):
            subgraph = graph.subgraph(nodes)
            
            if self._check_motif_properties(subgraph, motif):
                match_score = self._compute_match_score(subgraph, motif.pattern)
                
                if match_score >= min_score:
                    instances.append({
                        'nodes': list(nodes),
                        'score': match_score,
                        'motif': motif_name,
                        'properties': self._extract_properties(subgraph)
                    })
                    
        return instances
        
    def _check_size_compatibility(
        self,
        circuit: nx.DiGraph,
        motif: CircuitMotif
    ) -> bool:
        """Check if circuit size is compatible with motif."""
        circuit_size = circuit.number_of_nodes()
        return motif.min_size <= circuit_size <= motif.max_size
        
    def _compute_match_score(
        self,
        circuit: nx.DiGraph,
        pattern: nx.DiGraph
    ) -> float:
        """
        Compute structural similarity between circuit and pattern.
        """
        # Use graph isomorphism with node/edge attributes
        if circuit.number_of_nodes() < pattern.number_of_nodes():
            return 0.0
            
        # Try to find best subgraph match
        best_score = 0.0
        
        # For each possible node mapping
        for mapping in self._get_possible_mappings(circuit, pattern):
            score = self._score_mapping(circuit, pattern, mapping)
            best_score = max(best_score, score)
            
        return best_score
        
    def _check_motif_properties(
        self,
        subgraph: nx.DiGraph,
        motif: CircuitMotif
    ) -> bool:
        """Check if subgraph satisfies motif properties."""
        for prop, value in motif.required_properties.items():
            if prop == 'coherent':
                if value and not self._is_coherent(subgraph):
                    return False
            elif prop == 'cyclic':
                if value and not nx.is_directed_acyclic_graph(subgraph):
                    return False
            elif prop == 'linear':
                if value and not self._is_linear(subgraph):
                    return False
            elif prop == 'central_node':
                if value and not self._has_central_node(subgraph):
                    return False
            elif prop == 'bipartite':
                if value and not self._is_bipartite_like(subgraph):
                    return False
                    
        return True
        
    def _get_subgraphs(
        self,
        graph: nx.DiGraph,
        size: int
    ) -> List[Set]:
        """Get all connected subgraphs of given size."""
        subgraphs = []
        nodes = list(graph.nodes())
        
        # Use DFS to find connected subgraphs
        from itertools import combinations
        
        for node_subset in combinations(nodes, size):
            subgraph = graph.subgraph(node_subset)
            
            # Check if connected (weakly)
            if nx.is_weakly_connected(subgraph):
                subgraphs.append(set(node_subset))
                
        return subgraphs
        
    def _get_possible_mappings(
        self,
        graph1: nx.DiGraph,
        graph2: nx.DiGraph
    ) -> List[Dict]:
        """Get possible node mappings between graphs."""
        # Simplified - in practice use more sophisticated matching
        mappings = []
        
        if graph1.number_of_nodes() >= graph2.number_of_nodes():
            nodes1 = list(graph1.nodes())
            nodes2 = list(graph2.nodes())
            
            from itertools import permutations
            
            for perm in permutations(nodes1, len(nodes2)):
                mapping = dict(zip(nodes2, perm))
                mappings.append(mapping)
                
                if len(mappings) > 100:  # Limit for efficiency
                    break
                    
        return mappings
        
    def _score_mapping(
        self,
        graph1: nx.DiGraph,
        graph2: nx.DiGraph,
        mapping: Dict
    ) -> float:
        """Score a specific node mapping."""
        score = 0.0
        total_edges = graph2.number_of_edges()
        
        if total_edges == 0:
            return 0.0
            
        # Check edge preservation
        for u, v in graph2.edges():
            if mapping[u] in graph1 and mapping[v] in graph1:
                if graph1.has_edge(mapping[u], mapping[v]):
                    score += 1.0
                    
        return score / total_edges
        
    def _extract_properties(self, subgraph: nx.DiGraph) -> Dict:
        """Extract properties of a subgraph."""
        properties = {
            'num_nodes': subgraph.number_of_nodes(),
            'num_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph),
            'is_dag': nx.is_directed_acyclic_graph(subgraph),
            'max_in_degree': max(dict(subgraph.in_degree()).values()) if subgraph else 0,
            'max_out_degree': max(dict(subgraph.out_degree()).values()) if subgraph else 0
        }
        
        return properties
        
    def _is_coherent(self, graph: nx.DiGraph) -> bool:
        """Check if graph represents coherent feed-forward."""
        # Simplified check - all paths from input to output have same sign
        return True  # Placeholder
        
    def _is_linear(self, graph: nx.DiGraph) -> bool:
        """Check if graph is linear chain."""
        # Each node has at most one in and one out edge
        for node in graph.nodes():
            if graph.in_degree(node) > 1 or graph.out_degree(node) > 1:
                return False
        return True
        
    def _has_central_node(self, graph: nx.DiGraph) -> bool:
        """Check if graph has a central hub node."""
        degrees = dict(graph.degree())
        if not degrees:
            return False
            
        max_degree = max(degrees.values())
        avg_degree = sum(degrees.values()) / len(degrees)
        
        # Central node has much higher degree than average
        return max_degree > avg_degree * 2
        
    def _is_bipartite_like(self, graph: nx.DiGraph) -> bool:
        """Check if graph has bipartite-like structure."""
        # Simplified check
        undirected = graph.to_undirected()
        return nx.is_bipartite(undirected)


class MotifScorer:
    """
    Scores motifs based on various criteria.
    """
    
    def __init__(
        self,
        conservation_weight: float = 0.3,
        function_weight: float = 0.3,
        rarity_weight: float = 0.2,
        size_weight: float = 0.2
    ):
        self.conservation_weight = conservation_weight
        self.function_weight = function_weight
        self.rarity_weight = rarity_weight
        self.size_weight = size_weight
        
    def score_motif_instance(
        self,
        instance: Dict,
        motif: CircuitMotif,
        background_frequency: float,
        functional_annotation: Optional[str] = None
    ) -> float:
        """
        Score a specific motif instance.
        
        Args:
            instance: Motif instance dictionary
            motif: Motif definition
            background_frequency: Expected frequency by chance
            functional_annotation: Known function if available
            
        Returns:
            Overall motif score
        """
        # Conservation score from motif definition
        conservation_score = motif.conservation_score
        
        # Function score based on annotation match
        if functional_annotation and motif.biological_function:
            function_score = self._compute_function_similarity(
                functional_annotation,
                motif.biological_function
            )
        else:
            function_score = 0.5  # Neutral
            
        # Rarity score - how surprising is this motif
        if background_frequency > 0:
            rarity_score = 1.0 - background_frequency
        else:
            rarity_score = 0.9  # Very rare
            
        # Size score - prefer moderate sized motifs
        size = instance['properties']['num_nodes']
        optimal_size = (motif.min_size + motif.max_size) / 2
        size_deviation = abs(size - optimal_size) / optimal_size
        size_score = 1.0 - min(size_deviation, 1.0)
        
        # Combine scores
        total_score = (
            conservation_score * self.conservation_weight +
            function_score * self.function_weight +
            rarity_score * self.rarity_weight +
            size_score * self.size_weight
        )
        
        return total_score
        
    def rank_motifs(
        self,
        motif_instances: List[Dict],
        motif_library: MotifLibrary
    ) -> List[Tuple[Dict, float]]:
        """
        Rank motif instances by their scores.
        
        Args:
            motif_instances: List of motif instances
            motif_library: Library of motif definitions
            
        Returns:
            Sorted list of (instance, score) tuples
        """
        scored_instances = []
        
        # Compute background frequencies
        motif_counts = defaultdict(int)
        for instance in motif_instances:
            motif_counts[instance['motif']] += 1
            
        total_instances = len(motif_instances)
        
        for instance in motif_instances:
            motif_name = instance['motif']
            motif = motif_library.get_motif(motif_name)
            
            if motif:
                background_freq = motif_counts[motif_name] / (total_instances + 1)
                
                score = self.score_motif_instance(
                    instance,
                    motif,
                    background_freq
                )
                
                scored_instances.append((instance, score))
                
        # Sort by score
        scored_instances.sort(key=lambda x: x[1], reverse=True)
        
        return scored_instances
        
    def _compute_function_similarity(
        self,
        annotation1: str,
        annotation2: str
    ) -> float:
        """Compute similarity between functional annotations."""
        # Simplified - in practice use ontology-based similarity
        annotation1_lower = annotation1.lower()
        annotation2_lower = annotation2.lower()
        
        # Exact match
        if annotation1_lower == annotation2_lower:
            return 1.0
            
        # Partial match
        words1 = set(annotation1_lower.split())
        words2 = set(annotation2_lower.split())
        
        if words1 and words2:
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return overlap / union
            
        return 0.0