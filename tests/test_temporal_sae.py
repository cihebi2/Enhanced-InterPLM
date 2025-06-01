# tests/test_temporal_sae.py

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from enhanced_interplm.temporal_sae.temporal_autoencoder import (
    TemporalSAE, TemporalFeatureTracker
)


class TestTemporalSAE:
    """Test cases for Temporal SAE."""
    
    @pytest.fixture
    def model_config(self):
        return {
            'input_dim': 64,
            'hidden_dim': 128,
            'dict_size': 256,
            'num_layers': 4,
            'num_attention_heads': 4,
            'dropout': 0.1
        }
        
    @pytest.fixture
    def temporal_sae(self, model_config):
        return TemporalSAE(**model_config)
        
    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        num_layers = 4
        seq_len = 10
        embed_dim = 64
        
        # Random embeddings
        embeddings = torch.randn(batch_size, num_layers, seq_len, embed_dim)
        return embeddings
        
    def test_initialization(self, temporal_sae, model_config):
        """Test model initialization."""
        assert temporal_sae.input_dim == model_config['input_dim']
        assert temporal_sae.hidden_dim == model_config['hidden_dim']
        assert temporal_sae.dict_size == model_config['dict_size']
        assert temporal_sae.num_layers == model_config['num_layers']
        
    def test_forward_pass(self, temporal_sae, sample_data):
        """Test forward pass."""
        temporal_sae.eval()
        
        with torch.no_grad():
            reconstructed, features = temporal_sae(sample_data, return_features=True)
            
        # Check shapes
        assert reconstructed.shape == sample_data.shape
        assert features.shape[0] == sample_data.shape[0]  # batch size
        assert features.shape[1] == sample_data.shape[1]  # num layers
        assert features.shape[2] == sample_data.shape[2]  # seq len
        assert features.shape[3] == temporal_sae.dict_size
        
    def test_encode_decode(self, temporal_sae, sample_data):
        """Test encode and decode separately."""
        temporal_sae.eval()
        
        with torch.no_grad():
            # Encode
            features = temporal_sae.encode(sample_data)
            
            # Decode
            reconstructed = temporal_sae.decode(features)
            
        assert reconstructed.shape == sample_data.shape
        
    def test_circuit_importance(self, temporal_sae, sample_data):
        """Test circuit importance computation."""
        temporal_sae.eval()
        
        with torch.no_grad():
            _, features = temporal_sae(sample_data, return_features=True)
            importance = temporal_sae.compute_circuit_importance(features)
            
        # Check shape
        assert importance.shape == (
            temporal_sae.dict_size,
            temporal_sae.dict_size,
            temporal_sae.num_layers - 1
        )
        
        # Check values are in reasonable range
        assert importance.min() >= -1
        assert importance.max() <= 1
        
    def test_gradient_flow(self, temporal_sae, sample_data):
        """Test gradient flow through the model."""
        temporal_sae.train()
        
        # Forward pass
        reconstructed, features = temporal_sae(sample_data, return_features=True)
        
        # Compute loss
        loss = F.mse_loss(reconstructed, sample_data) + features.abs().mean() * 0.1
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in temporal_sae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                
    def test_time_decay_weights(self, temporal_sae):
        """Test time decay weight computation."""
        weights = temporal_sae.time_decay_weights
        
        # Check shape
        assert weights.shape == (temporal_sae.num_layers, temporal_sae.num_layers)
        
        # Check diagonal is 1
        assert torch.allclose(weights.diag(), torch.ones(temporal_sae.num_layers))
        
        # Check decay pattern
        for i in range(temporal_sae.num_layers):
            for j in range(i):
                expected = temporal_sae.time_decay_factor ** (i - j)
                assert torch.allclose(weights[i, j], torch.tensor(expected))
                

class TestTemporalFeatureTracker:
    """Test cases for Temporal Feature Tracker."""
    
    @pytest.fixture
    def tracker(self):
        return TemporalFeatureTracker(feature_dim=128, num_layers=4)
        
    @pytest.fixture
    def sample_features(self):
        batch_size = 2
        num_layers = 4
        seq_len = 10
        feature_dim = 128
        
        return torch.randn(batch_size, num_layers, seq_len, feature_dim)
        
    def test_feature_evolution_tracking(self, tracker, sample_features):
        """Test feature evolution tracking."""
        tracker.eval()
        
        with torch.no_grad():
            evolution_metrics = tracker.track_feature_evolution(sample_features)
            
        # Check all metrics are present
        assert 'feature_stability' in evolution_metrics
        assert 'feature_emergence' in evolution_metrics
        assert 'feature_decay' in evolution_metrics
        
        # Check shapes
        seq_len = sample_features.shape[2]
        num_layers = sample_features.shape[1]
        
        assert evolution_metrics['feature_stability'].shape == (num_layers - 1, seq_len)
        assert evolution_metrics['feature_emergence'].shape == (num_layers - 1, seq_len)
        assert evolution_metrics['feature_decay'].shape == (num_layers - 1, seq_len)
        
    def test_causal_strength(self, tracker, sample_features):
        """Test causal strength computation."""
        tracker.eval()
        
        source_features = sample_features[:, 0]  # First layer
        target_features = sample_features[:, 1]  # Second layer
        
        with torch.no_grad():
            causal_matrix = tracker.compute_causal_strength(
                source_features, target_features
            )
            
        # Check shape
        feature_dim = sample_features.shape[-1]
        assert causal_matrix.shape == (feature_dim, feature_dim)
        
        # Check values are in [0, 1] (sigmoid output)
        assert causal_matrix.min() >= 0
        assert causal_matrix.max() <= 1


# tests/test_circuit_discovery.py

import pytest
import torch
import networkx as nx
from enhanced_interplm.circuit_discovery.graph_builder import (
    DynamicGraphBuilder, CircuitMotifDetector, CircuitValidator
)


class TestDynamicGraphBuilder:
    """Test cases for Dynamic Graph Builder."""
    
    @pytest.fixture
    def graph_builder(self):
        return DynamicGraphBuilder(
            mutual_info_threshold=0.1,
            causal_threshold=0.3
        )
        
    @pytest.fixture
    def sample_features(self):
        # Create feature dict for 3 layers
        features = {}
        batch_size = 4
        seq_len = 20
        feature_dim = 64
        
        for layer in range(3):
            features[layer] = torch.randn(batch_size, seq_len, feature_dim)
            
        return features
        
    def test_graph_construction(self, graph_builder, sample_features):
        """Test feature interaction graph construction."""
        graph = graph_builder.build_feature_interaction_graph(sample_features)
        
        # Check graph properties
        assert isinstance(graph, nx.DiGraph)
        
        # Check nodes
        expected_nodes = []
        for layer in range(3):
            for feat in range(sample_features[0].shape[-1]):
                expected_nodes.append(f"L{layer}_F{feat}")
                
        assert len(graph.nodes) == len(expected_nodes)
        
        # Check edges exist
        assert graph.number_of_edges() > 0
        
        # Check edge attributes
        for u, v, data in graph.edges(data=True):
            assert 'weight' in data
            assert 'mutual_info' in data
            assert 'correlation' in data
            
            # Check value ranges
            assert 0 <= data['weight'] <= 1
            assert data['mutual_info'] >= 0
            assert -1 <= data['correlation'] <= 1
            
    def test_feature_interactions(self, graph_builder):
        """Test feature interaction computation."""
        # Simple test case
        features_1 = torch.randn(100, 10)
        features_2 = torch.randn(100, 10)
        
        interactions = graph_builder._compute_feature_interactions(
            features_1.unsqueeze(1),  # Add seq_len dimension
            features_2.unsqueeze(1),
            same_layer=False
        )
        
        # Check all pairs are computed
        assert len(interactions) == 10 * 10
        
        # Check interaction metrics
        for (i, j), metrics in interactions.items():
            assert 'mutual_info' in metrics
            assert 'correlation' in metrics
            assert 'causal_strength' in metrics
            
            assert metrics['mutual_info'] >= 0
            assert -1 <= metrics['correlation'] <= 1
            assert 0 <= metrics['causal_strength'] <= 1


class TestCircuitMotifDetector:
    """Test cases for Circuit Motif Detector."""
    
    @pytest.fixture
    def motif_detector(self):
        return CircuitMotifDetector(min_motif_size=3, max_motif_size=5)
        
    @pytest.fixture
    def sample_graph(self):
        # Create a simple graph with known motifs
        graph = nx.DiGraph()
        
        # Add a triangle motif
        graph.add_edges_from([
            ('L0_F0', 'L1_F0', {'weight': 0.8}),
            ('L1_F0', 'L2_F0', {'weight': 0.7}),
            ('L0_F0', 'L2_F0', {'weight': 0.6}),
        ])
        
        # Add a chain motif
        graph.add_edges_from([
            ('L0_F1', 'L1_F1', {'weight': 0.9}),
            ('L1_F1', 'L2_F1', {'weight': 0.8}),
        ])
        
        return graph
        
    def test_motif_detection(self, motif_detector, sample_graph):
        """Test circuit motif detection."""
        motifs = motif_detector.find_circuit_motifs(sample_graph, min_frequency=1)
        
        # Should find at least some motifs
        assert len(motifs) > 0
        
        # Check motif structure
        for motif in motifs:
            assert 'structure' in motif
            assert 'instances' in motif
            assert 'frequency' in motif
            assert 'avg_weight' in motif
            assert 'importance' in motif
            
            # Check frequency matches instances
            assert motif['frequency'] == len(motif['instances'])
            
    def test_motif_ranking(self, motif_detector):
        """Test motif ranking."""
        # Create mock motifs
        motifs = [
            {'frequency': 5, 'avg_weight': 0.8, 'importance': 0},
            {'frequency': 10, 'avg_weight': 0.6, 'importance': 0},
            {'frequency': 3, 'avg_weight': 0.9, 'importance': 0},
        ]
        
        ranked = motif_detector._rank_motifs(motifs)
        
        # Check importance scores are computed
        for motif in ranked:
            assert motif['importance'] > 0
            
        # Check ordering (descending by importance)
        importances = [m['importance'] for m in ranked]
        assert importances == sorted(importances, reverse=True)


class TestCircuitValidator:
    """Test cases for Circuit Validator."""
    
    @pytest.fixture
    def validator(self):
        return CircuitValidator(validation_threshold=0.5)
        
    @pytest.fixture
    def sample_circuits(self):
        return [
            {
                'instances': [['L0_F0', 'L1_F0', 'L2_F0']],
                'frequency': 3,
                'importance': 0.8
            },
            {
                'instances': [['L0_F1', 'L1_F1']],
                'frequency': 2,
                'importance': 0.6
            }
        ]
        
    @pytest.fixture
    def sample_features(self):
        features = {}
        for layer in range(3):
            features[layer] = torch.randn(4, 20, 64)
        return features
        
    def test_circuit_validation(self, validator, sample_circuits, sample_features):
        """Test circuit validation."""
        validated = validator.validate_circuits(
            sample_circuits, sample_features
        )
        
        # Check validation adds scores
        for circuit in validated:
            assert 'validation_scores' in circuit
            assert 'overall_score' in circuit
            
            scores = circuit['validation_scores']
            assert 'coherence' in scores
            assert 'stability' in scores
            assert 'task_relevance' in scores
            
            # Check score ranges
            assert 0 <= scores['coherence'] <= 1
            assert 0 <= scores['stability'] <= 1
            assert scores['task_relevance'] >= 0
            
    def test_coherence_computation(self, validator, sample_circuits, sample_features):
        """Test coherence score computation."""
        circuit = sample_circuits[0]
        
        coherence = validator._compute_coherence_score(circuit, sample_features)
        
        # Check coherence is in valid range
        assert -1 <= coherence <= 1


# tests/test_biophysics.py

import pytest
import torch
from enhanced_interplm.biophysics.physics_constraints import (
    BiophysicsConstraintModule, BiophysicsGuidedSAE, AminoAcidProperties
)


class TestBiophysicsConstraintModule:
    """Test cases for Biophysics Constraint Module."""
    
    @pytest.fixture
    def physics_module(self):
        return BiophysicsConstraintModule(
            feature_dim=128,
            hidden_dim=256,
            constraint_weight=0.1
        )
        
    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        seq_len = 50
        feature_dim = 128
        
        features = torch.randn(batch_size, seq_len, feature_dim)
        sequences = ['ACDEFGHIKLMNPQRSTVWY' * 3][:seq_len]
        sequences = [sequences, sequences]  # Two sequences
        structures = torch.randn(batch_size, seq_len, 3) * 10  # Random coords
        
        return features, sequences, structures
        
    def test_forward_pass(self, physics_module, sample_data):
        """Test forward pass with constraints."""
        features, sequences, structures = sample_data
        physics_module.eval()
        
        with torch.no_grad():
            constrained_features, losses = physics_module(
                features, sequences, structures
            )
            
        # Check output shape
        assert constrained_features.shape == features.shape
        
        # Check losses
        assert 'hydrophobic' in losses
        assert 'charge' in losses
        assert 'hbond' in losses
        assert 'spatial' in losses
        
        # Check loss values are reasonable
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.item() >= 0
            assert not torch.isnan(loss_value)
            
    def test_hydrophobicity_constraint(self, physics_module):
        """Test hydrophobicity constraint loss."""
        batch_size = 1
        seq_len = 10
        
        # Create features and sequence
        features = torch.randn(batch_size, seq_len, 128)
        sequences = ['IIIVVVLLLL']  # All hydrophobic
        
        # Get hydrophobicity predictions
        hydro_pred = physics_module.hydrophobic_extractor(features)
        
        # Compute loss
        loss = physics_module._hydrophobic_constraint_loss(hydro_pred, sequences)
        
        assert loss.item() >= 0
        
    def test_charge_constraint(self, physics_module):
        """Test charge constraint loss."""
        batch_size = 1
        seq_len = 10
        
        features = torch.randn(batch_size, seq_len, 128)
        sequences = ['KRKRKDEDED']  # Mixed charges
        
        charge_pred = physics_module.charge_extractor(features)
        loss = physics_module._charge_constraint_loss(charge_pred, sequences)
        
        assert loss.item() >= 0
        
    def test_spatial_constraint(self, physics_module, sample_data):
        """Test spatial constraint loss."""
        features, sequences, structures = sample_data
        
        physics_module.eval()
        with torch.no_grad():
            loss = physics_module._spatial_constraint_loss(
                features, structures, sequences
            )
            
        assert loss.item() >= 0


class TestBiophysicsGuidedSAE:
    """Test cases for Biophysics-Guided SAE."""
    
    @pytest.fixture
    def bio_sae(self):
        return BiophysicsGuidedSAE(
            activation_dim=128,
            dict_size=512,
            hidden_dim=256,
            physics_weight=0.1
        )
        
    @pytest.fixture
    def sample_input(self):
        batch_size = 2
        seq_len = 30
        activation_dim = 128
        
        x = torch.randn(batch_size, seq_len, activation_dim)
        sequences = ['ACDEFGHIKLMNPQRSTVWY'][:seq_len]
        sequences = [sequences] * batch_size
        structures = torch.randn(batch_size, seq_len, 3) * 10
        
        return x, sequences, structures
        
    def test_forward_pass(self, bio_sae, sample_input):
        """Test forward pass."""
        x, sequences, structures = sample_input
        bio_sae.eval()
        
        with torch.no_grad():
            reconstructed, features, physics_losses = bio_sae(
                x, sequences, structures, return_physics_loss=True
            )
            
        # Check shapes
        assert reconstructed.shape == x.shape
        assert features.shape[:-1] == x.shape[:-1]
        assert features.shape[-1] == bio_sae.dict_size
        
        # Check physics losses
        assert physics_losses is not None
        assert all(isinstance(v, torch.Tensor) for v in physics_losses.values())
        
    def test_without_physics(self, bio_sae, sample_input):
        """Test forward pass without physics constraints."""
        x, _, _ = sample_input
        bio_sae.eval()
        
        with torch.no_grad():
            reconstructed, features, physics_losses = bio_sae(
                x, return_physics_loss=False
            )
            
        assert reconstructed.shape == x.shape
        assert features.shape[-1] == bio_sae.dict_size
        assert physics_losses is None
        
    def test_encode_decode(self, bio_sae, sample_input):
        """Test encode and decode separately."""
        x, _, _ = sample_input
        bio_sae.eval()
        
        with torch.no_grad():
            # Encode
            features = bio_sae.encode(x)
            
            # Decode
            reconstructed = bio_sae.decode(features)
            
        assert features.shape[-1] == bio_sae.dict_size
        assert reconstructed.shape == x.shape