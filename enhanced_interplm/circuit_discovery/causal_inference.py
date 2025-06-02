# 因果推断引擎 
# enhanced_interplm/circuit_discovery/causal_inference.py

"""
Causal inference engine for understanding feature relationships in circuits.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CausalRelation:
    """Represents a causal relationship between features."""
    cause_feature: int
    effect_feature: int
    causal_strength: float
    confidence: float
    lag: int = 0
    mechanism: str = 'direct'  # 'direct', 'mediated', 'confounded'
    mediators: Optional[List[int]] = None
    

class CausalInferenceEngine:
    """
    Engine for inferring causal relationships between features.
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_causal_strength: float = 0.3,
        max_lag: int = 3,
        use_nonlinear: bool = True
    ):
        self.significance_level = significance_level
        self.min_causal_strength = min_causal_strength
        self.max_lag = max_lag
        self.use_nonlinear = use_nonlinear
        
        # Store discovered relations
        self.causal_relations: List[CausalRelation] = []
        self.causal_graph: Optional[nx.DiGraph] = None
        
    def infer_causality(
        self,
        features: torch.Tensor,
        layer_indices: Optional[torch.Tensor] = None,
        known_confounders: Optional[Set[int]] = None
    ) -> Dict[str, Any]:
        """
        Infer causal relationships between features.
        
        Args:
            features: Feature tensor [batch_size, seq_len, num_features]
            layer_indices: Optional layer indices for temporal causality
            known_confounders: Set of feature indices that are known confounders
            
        Returns:
            Dictionary containing causal analysis results
        """
        self.causal_relations = []
        num_features = features.shape[-1]
        
        # Flatten features for analysis
        features_flat = features.reshape(-1, num_features).cpu().numpy()
        
        # 1. Pairwise causal discovery
        pairwise_relations = self._discover_pairwise_causality(features_flat)
        
        # 2. Test for mediation
        mediated_relations = self._test_mediation(features_flat, pairwise_relations)
        
        # 3. Handle confounders if known
        if known_confounders:
            pairwise_relations = self._adjust_for_confounders(
                features_flat, pairwise_relations, known_confounders
            )
            
        # 4. Temporal causality if layer indices provided
        if layer_indices is not None:
            temporal_relations = self._discover_temporal_causality(
                features, layer_indices
            )
            self.causal_relations.extend(temporal_relations)
            
        # 5. Build causal graph
        self.causal_graph = self._build_causal_graph(
            self.causal_relations + pairwise_relations + mediated_relations
        )
        
        # 6. Identify causal chains and feedback loops
        chains = self._find_causal_chains()
        loops = self._find_feedback_loops()
        
        # 7. Compute causal importance scores
        importance_scores = self._compute_causal_importance()
        
        return {
            'relations': self.causal_relations,
            'graph': self.causal_graph,
            'chains': chains,
            'loops': loops,
            'importance_scores': importance_scores,
            'summary': self._summarize_causality()
        }
        
    def _discover_pairwise_causality(
        self,
        features: np.ndarray
    ) -> List[CausalRelation]:
        """Discover pairwise causal relationships."""
        relations = []
        num_features = features.shape[1]
        
        for i in range(num_features):
            for j in range(num_features):
                if i == j:
                    continue
                    
                # Test multiple causal criteria
                causal_tests = []
                
                # 1. Granger causality test
                granger_score = self._granger_causality_test(
                    features[:, i], features[:, j]
                )
                causal_tests.append(granger_score)
                
                # 2. Transfer entropy
                if self.use_nonlinear:
                    te_score = self._transfer_entropy(
                        features[:, i], features[:, j]
                    )
                    causal_tests.append(te_score)
                    
                # 3. Conditional independence test
                ci_score = self._conditional_independence_test(
                    features[:, i], features[:, j], features
                )
                causal_tests.append(ci_score)
                
                # Combine scores
                causal_strength = np.mean(causal_tests)
                confidence = self._compute_confidence(causal_tests)
                
                if causal_strength > self.min_causal_strength:
                    relations.append(CausalRelation(
                        cause_feature=i,
                        effect_feature=j,
                        causal_strength=causal_strength,
                        confidence=confidence,
                        mechanism='direct'
                    ))
                    
        return relations
        
    def _granger_causality_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        max_lag: Optional[int] = None
    ) -> float:
        """Perform Granger causality test."""
        if max_lag is None:
            max_lag = self.max_lag
            
        # Simple implementation using linear regression
        # For production, use statsmodels.tsa.stattools.grangercausalitytests
        
        n = len(x)
        if n < max_lag + 10:  # Need sufficient data
            return 0.0
            
        best_score = 0.0
        
        for lag in range(1, max_lag + 1):
            # Prepare lagged data
            x_lag = x[:-lag]
            y_lag = y[:-lag]
            y_current = y[lag:]
            
            # Fit models
            # Model 1: y_t ~ y_{t-1}
            coef1 = np.corrcoef(y_lag, y_current)[0, 1]
            
            # Model 2: y_t ~ y_{t-1} + x_{t-1}
            # Use multiple regression approximation
            X = np.column_stack([y_lag, x_lag])
            # Simple least squares
            coef = np.linalg.lstsq(X, y_current, rcond=None)[0]
            
            # Improvement in prediction
            pred1 = coef1 * y_lag
            pred2 = X @ coef
            
            mse1 = np.mean((y_current - pred1) ** 2)
            mse2 = np.mean((y_current - pred2) ** 2)
            
            # F-statistic approximation
            if mse2 < mse1:
                f_stat = (mse1 - mse2) / mse2 * (n - lag - 2)
                p_value = 1 - stats.f.cdf(f_stat, 1, n - lag - 2)
                
                if p_value < self.significance_level:
                    score = 1 - p_value
                    best_score = max(best_score, score)
                    
        return best_score
        
    def _transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int = 10
    ) -> float:
        """Compute transfer entropy from x to y."""
        # Discretize continuous values
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        
        x_disc = discretizer.fit_transform(x.reshape(-1, 1)).flatten()
        y_disc = discretizer.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Compute transfer entropy
        # TE(X->Y) = I(Y_t; X_{t-1} | Y_{t-1})
        
        n = len(x_disc) - 1
        
        # Joint probabilities
        xyz_hist = np.zeros((bins, bins, bins))
        yz_hist = np.zeros((bins, bins))
        y_hist = np.zeros(bins)
        
        for i in range(n):
            yt = int(y_disc[i+1])
            yt_1 = int(y_disc[i])
            xt_1 = int(x_disc[i])
            
            xyz_hist[yt, xt_1, yt_1] += 1
            yz_hist[yt, yt_1] += 1
            y_hist[yt] += 1
            
        # Normalize
        xyz_hist = xyz_hist / n
        yz_hist = yz_hist / n
        y_hist = y_hist / n
        
        # Compute TE
        te = 0.0
        for yt in range(bins):
            for xt_1 in range(bins):
                for yt_1 in range(bins):
                    if xyz_hist[yt, xt_1, yt_1] > 0 and yz_hist[yt, yt_1] > 0:
                        te += xyz_hist[yt, xt_1, yt_1] * np.log(
                            xyz_hist[yt, xt_1, yt_1] * y_hist[yt] /
                            (yz_hist[yt, yt_1] * xyz_hist[:, xt_1, yt_1].sum() + 1e-10)
                        )
                        
        return max(0, te)
        
    def _conditional_independence_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray
    ) -> float:
        """Test conditional independence of x and y given z."""
        # Use partial correlation as a simple test
        # For more sophisticated tests, use PC algorithm or kernel methods
        
        # Remove influence of other variables
        z_others = np.delete(z, [x.argmax(), y.argmax()], axis=1)
        
        if z_others.shape[1] == 0:
            # No conditioning variables
            return abs(np.corrcoef(x, y)[0, 1])
            
        # Regress out other variables
        x_resid = x - np.mean(x)
        y_resid = y - np.mean(y)
        
        for i in range(min(5, z_others.shape[1])):  # Limit conditioning set
            z_var = z_others[:, i]
            if np.std(z_var) > 0:
                x_resid = x_resid - np.corrcoef(x_resid, z_var)[0, 1] * z_var
                y_resid = y_resid - np.corrcoef(y_resid, z_var)[0, 1] * z_var
                
        # Compute partial correlation
        if np.std(x_resid) > 0 and np.std(y_resid) > 0:
            partial_corr = np.corrcoef(x_resid, y_resid)[0, 1]
            return abs(partial_corr)
        else:
            return 0.0
            
    def _test_mediation(
        self,
        features: np.ndarray,
        direct_relations: List[CausalRelation]
    ) -> List[CausalRelation]:
        """Test for mediated causal relationships."""
        mediated_relations = []
        
        # Build adjacency matrix from direct relations
        num_features = features.shape[1]
        adj_matrix = np.zeros((num_features, num_features))
        
        for rel in direct_relations:
            adj_matrix[rel.cause_feature, rel.effect_feature] = rel.causal_strength
            
        # Test each potential mediation path
        for i in range(num_features):
            for j in range(num_features):
                if i == j or adj_matrix[i, j] > 0:
                    continue
                    
                # Find potential mediators
                for k in range(num_features):
                    if k == i or k == j:
                        continue
                        
                    # Check if i -> k -> j path exists
                    if adj_matrix[i, k] > 0 and adj_matrix[k, j] > 0:
                        # Test mediation using Sobel test approximation
                        x = features[:, i]
                        m = features[:, k]
                        y = features[:, j]
                        
                        # Path coefficients
                        a = np.corrcoef(x, m)[0, 1]  # X -> M
                        b = np.corrcoef(m, y)[0, 1]  # M -> Y
                        c = np.corrcoef(x, y)[0, 1]  # X -> Y (total)
                        
                        # Compute c' (direct effect controlling for M)
                        if np.std(m) > 0:
                            x_resid = x - np.mean(x) - a * (m - np.mean(m))
                            y_resid = y - np.mean(y) - b * (m - np.mean(m))
                            
                            if np.std(x_resid) > 0 and np.std(y_resid) > 0:
                                c_prime = np.corrcoef(x_resid, y_resid)[0, 1]
                            else:
                                c_prime = 0
                        else:
                            c_prime = c
                            
                        # Mediation effect
                        mediation_effect = c - c_prime
                        
                        if abs(mediation_effect) > self.min_causal_strength:
                            mediated_relations.append(CausalRelation(
                                cause_feature=i,
                                effect_feature=j,
                                causal_strength=abs(mediation_effect),
                                confidence=min(adj_matrix[i, k], adj_matrix[k, j]),
                                mechanism='mediated',
                                mediators=[k]
                            ))
                            
        return mediated_relations
        
    def _adjust_for_confounders(
        self,
        features: np.ndarray,
        relations: List[CausalRelation],
        confounders: Set[int]
    ) -> List[CausalRelation]:
        """Adjust causal estimates for known confounders."""
        adjusted_relations = []
        
        for rel in relations:
            # Skip if involves confounder
            if rel.cause_feature in confounders or rel.effect_feature in confounders:
                continue
                
            # Adjust for confounders
            x = features[:, rel.cause_feature]
            y = features[:, rel.effect_feature]
            
            # Regress out confounder effects
            x_adjusted = x.copy()
            y_adjusted = y.copy()
            
            for conf_idx in confounders:
                conf = features[:, conf_idx]
                if np.std(conf) > 0:
                    # Remove confounder influence
                    x_adjusted = x_adjusted - np.corrcoef(x_adjusted, conf)[0, 1] * conf
                    y_adjusted = y_adjusted - np.corrcoef(y_adjusted, conf)[0, 1] * conf
                    
            # Recompute causal strength
            if np.std(x_adjusted) > 0 and np.std(y_adjusted) > 0:
                adjusted_strength = abs(np.corrcoef(x_adjusted, y_adjusted)[0, 1])
                
                if adjusted_strength > self.min_causal_strength:
                    adjusted_rel = CausalRelation(
                        cause_feature=rel.cause_feature,
                        effect_feature=rel.effect_feature,
                        causal_strength=adjusted_strength,
                        confidence=rel.confidence * 0.9,  # Reduce confidence
                        mechanism='adjusted',
                        mediators=list(confounders)
                    )
                    adjusted_relations.append(adjusted_rel)
                    
        return adjusted_relations
        
    def _discover_temporal_causality(
        self,
        features: torch.Tensor,
        layer_indices: torch.Tensor
    ) -> List[CausalRelation]:
        """Discover temporal causal relationships across layers."""
        temporal_relations = []
        num_layers = layer_indices.max().item() + 1
        num_features = features.shape[-1]
        
        # Analyze causality between consecutive layers
        for layer in range(num_layers - 1):
            # Get features for current and next layer
            curr_mask = layer_indices == layer
            next_mask = layer_indices == layer + 1
            
            if curr_mask.sum() == 0 or next_mask.sum() == 0:
                continue
                
            curr_features = features[curr_mask].mean(dim=0).cpu().numpy()
            next_features = features[next_mask].mean(dim=0).cpu().numpy()
            
            # Test causality for each feature pair
            for i in range(num_features):
                for j in range(num_features):
                    # Compute temporal correlation
                    if curr_features[:, i].std() > 0 and next_features[:, j].std() > 0:
                        temp_corr = np.corrcoef(curr_features[:, i], next_features[:, j])[0, 1]
                        
                        if abs(temp_corr) > self.min_causal_strength:
                            temporal_relations.append(CausalRelation(
                                cause_feature=i,
                                effect_feature=j,
                                causal_strength=abs(temp_corr),
                                confidence=0.8,  # Temporal relations have lower confidence
                                lag=1,
                                mechanism='temporal'
                            ))
                            
        return temporal_relations
        
    def _build_causal_graph(
        self,
        relations: List[CausalRelation]
    ) -> nx.DiGraph:
        """Build directed graph from causal relations."""
        graph = nx.DiGraph()
        
        # Add edges with attributes
        for rel in relations:
            graph.add_edge(
                rel.cause_feature,
                rel.effect_feature,
                weight=rel.causal_strength,
                confidence=rel.confidence,
                mechanism=rel.mechanism,
                lag=rel.lag
            )
            
        return graph
        
    def _find_causal_chains(self) -> List[List[int]]:
        """Find causal chains in the graph."""
        if self.causal_graph is None:
            return []
            
        chains = []
        
        # Find all simple paths
        for source in self.causal_graph.nodes():
            for target in self.causal_graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.causal_graph, source, target, cutoff=5
                        ))
                        
                        for path in paths:
                            if len(path) >= 3:  # At least one intermediate node
                                chains.append(path)
                    except nx.NetworkXNoPath:
                        continue
                        
        # Remove duplicate chains
        unique_chains = []
        for chain in chains:
            if chain not in unique_chains:
                unique_chains.append(chain)
                
        return unique_chains
        
    def _find_feedback_loops(self) -> List[List[int]]:
        """Find feedback loops in the causal graph."""
        if self.causal_graph is None:
            return []
            
        # Find all simple cycles
        try:
            cycles = list(nx.simple_cycles(self.causal_graph))
            return [cycle for cycle in cycles if len(cycle) >= 2]
        except:
            return []
            
    def _compute_causal_importance(self) -> Dict[int, float]:
        """Compute importance scores based on causal structure."""
        if self.causal_graph is None:
            return {}
            
        importance = {}
        
        # PageRank-based importance
        try:
            pagerank = nx.pagerank(self.causal_graph, weight='weight')
            importance.update(pagerank)
        except:
            pass
            
        # Betweenness centrality
        try:
            betweenness = nx.betweenness_centrality(self.causal_graph, weight='weight')
            for node, score in betweenness.items():
                importance[node] = importance.get(node, 0) + score
        except:
            pass
            
        # In-degree and out-degree importance
        for node in self.causal_graph.nodes():
            in_degree = self.causal_graph.in_degree(node, weight='weight')
            out_degree = self.causal_graph.out_degree(node, weight='weight')
            
            importance[node] = importance.get(node, 0) + (in_degree + out_degree) / 2
            
        # Normalize scores
        if importance:
            max_score = max(importance.values())
            if max_score > 0:
                importance = {k: v / max_score for k, v in importance.items()}
                
        return importance
        
    def _compute_confidence(self, scores: List[float]) -> float:
        """Compute confidence from multiple test scores."""
        if not scores:
            return 0.0
            
        # Use harmonic mean to penalize inconsistent results
        scores = [s for s in scores if s > 0]
        if not scores:
            return 0.0
            
        harmonic_mean = len(scores) / sum(1/s for s in scores)
        
        # Adjust by consistency
        consistency = 1 - np.std(scores) / (np.mean(scores) + 1e-10)
        
        return harmonic_mean * consistency
        
    def _summarize_causality(self) -> Dict[str, Any]:
        """Summarize causal analysis results."""
        if not self.causal_relations:
            return {'status': 'No causal relations found'}
            
        summary = {
            'num_relations': len(self.causal_relations),
            'num_direct': sum(1 for r in self.causal_relations if r.mechanism == 'direct'),
            'num_mediated': sum(1 for r in self.causal_relations if r.mechanism == 'mediated'),
            'num_temporal': sum(1 for r in self.causal_relations if r.mechanism == 'temporal'),
            'avg_strength': np.mean([r.causal_strength for r in self.causal_relations]),
            'avg_confidence': np.mean([r.confidence for r in self.causal_relations])
        }
        
        # Find most influential features
        if self.causal_graph:
            out_degrees = dict(self.causal_graph.out_degree(weight='weight'))
            in_degrees = dict(self.causal_graph.in_degree(weight='weight'))
            
            summary['top_causes'] = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            summary['top_effects'] = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
        return summary