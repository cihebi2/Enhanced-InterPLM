# enhanced_interplm/train_enhanced_sae.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import wandb
from tqdm import tqdm
import json

from enhanced_interplm.temporal_sae.temporal_autoencoder import TemporalSAE, TemporalFeatureTracker
from enhanced_interplm.circuit_discovery.graph_builder import DynamicGraphBuilder, CircuitMotifDetector, CircuitValidator
from enhanced_interplm.biophysics.physics_constraints import BiophysicsGuidedSAE
from enhanced_interplm.hierarchical_mapping.hierarchical_mapper import (
    AminoAcidPropertyMapper, SecondaryStructureMapper, 
    DomainFunctionMapper, CrossLevelIntegrator
)


class EnhancedSAETrainer:
    """
    Comprehensive trainer for the enhanced SAE with all innovative components.
    """
    
    def __init__(
        self,
        config: Dict,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._initialize_models()
        
        # Initialize optimizers
        self._initialize_optimizers()
        
        # Initialize analysis components
        self.graph_builder = DynamicGraphBuilder()
        self.motif_detector = CircuitMotifDetector()
        self.circuit_validator = CircuitValidator()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _initialize_models(self):
        """Initialize all model components."""
        config = self.config
        
        # Temporal SAE
        self.temporal_sae = TemporalSAE(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            dict_size=config['dict_size'],
            num_layers=config['num_layers'],
            num_attention_heads=config.get('num_attention_heads', 8),
            dropout=config.get('dropout', 0.1)
        ).to(self.device)
        
        # Feature tracker
        self.feature_tracker = TemporalFeatureTracker(
            feature_dim=config['dict_size'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        # Biophysics-guided SAE
        self.biophysics_sae = BiophysicsGuidedSAE(
            activation_dim=config['input_dim'],
            dict_size=config['dict_size'],
            physics_weight=config.get('physics_weight', 0.1)
        ).to(self.device)
        
        # Hierarchical mappers
        self.aa_mapper = AminoAcidPropertyMapper(
            feature_dim=config['dict_size']
        ).to(self.device)
        
        self.ss_mapper = SecondaryStructureMapper(
            feature_dim=config['dict_size']
        ).to(self.device)
        
        self.domain_mapper = DomainFunctionMapper(
            feature_dim=config['dict_size']
        ).to(self.device)
        
        self.cross_level_integrator = CrossLevelIntegrator().to(self.device)
        
    def _initialize_optimizers(self):
        """Initialize optimizers for all components."""
        # Main optimizer for temporal SAE
        self.temporal_optimizer = optim.AdamW(
            self.temporal_sae.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Optimizer for biophysics SAE
        self.biophysics_optimizer = optim.AdamW(
            self.biophysics_sae.parameters(),
            lr=self.config['learning_rate'] * 0.5
        )
        
        # Optimizers for hierarchical mappers
        self.mapper_optimizers = {
            'aa': optim.Adam(self.aa_mapper.parameters(), lr=self.config['learning_rate'] * 0.1),
            'ss': optim.Adam(self.ss_mapper.parameters(), lr=self.config['learning_rate'] * 0.1),
            'domain': optim.Adam(self.domain_mapper.parameters(), lr=self.config['learning_rate'] * 0.1)
        }
        
        # Learning rate schedulers
        self.temporal_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.temporal_optimizer,
            T_max=self.config['num_epochs']
        )
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.temporal_sae.train()
        self.biophysics_sae.train()
        
        epoch_losses = {
            'temporal_recon': 0,
            'temporal_sparsity': 0,
            'biophysics_recon': 0,
            'biophysics_constraints': 0,
            'total': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data
            embeddings_sequence = batch['embeddings'].to(self.device)  # [B, L, S, D]
            sequences = batch['sequences']
            structures = batch.get('structures', None)
            
            if structures is not None:
                structures = structures.to(self.device)
                
            # Forward pass through temporal SAE
            reconstructed, features = self.temporal_sae(
                embeddings_sequence,
                return_features=True
            )
            
            # Compute temporal losses
            temporal_recon_loss = F.mse_loss(reconstructed, embeddings_sequence)
            temporal_sparsity_loss = features.abs().mean() * self.config['sparsity_weight']
            
            # Forward pass through biophysics SAE (last layer)
            last_layer_embeds = embeddings_sequence[:, -1]  # [B, S, D]
            bio_recon, bio_features, physics_losses = self.biophysics_sae(
                last_layer_embeds,
                sequences=sequences,
                structures=structures,
                return_physics_loss=True
            )
            
            # Compute biophysics losses
            bio_recon_loss = F.mse_loss(bio_recon, last_layer_embeds)
            bio_constraint_loss = sum(physics_losses.values()) if physics_losses else 0
            
            # Total loss
            total_loss = (
                temporal_recon_loss + 
                temporal_sparsity_loss + 
                bio_recon_loss * 0.5 + 
                bio_constraint_loss * self.config.get('physics_weight', 0.1)
            )
            
            # Backward pass
            self.temporal_optimizer.zero_grad()
            self.biophysics_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.temporal_sae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.biophysics_sae.parameters(), 1.0)
            
            self.temporal_optimizer.step()
            self.biophysics_optimizer.step()
            
            # Update epoch losses
            epoch_losses['temporal_recon'] += temporal_recon_loss.item()
            epoch_losses['temporal_sparsity'] += temporal_sparsity_loss.item()
            epoch_losses['biophysics_recon'] += bio_recon_loss.item()
            epoch_losses['biophysics_constraints'] += bio_constraint_loss.item() if isinstance(bio_constraint_loss, torch.Tensor) else bio_constraint_loss
            epoch_losses['total'] += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'recon': temporal_recon_loss.item()
            })
            
            self.global_step += 1
            
            # Periodic circuit analysis
            if self.global_step % self.config.get('circuit_analysis_freq', 100) == 0:
                self._analyze_circuits(features)
                
        # Average epoch losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.temporal_sae.eval()
        self.biophysics_sae.eval()
        
        eval_metrics = {
            'temporal_recon': 0,
            'feature_sparsity': 0,
            'circuit_coherence': 0,
            'hierarchical_consistency': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                embeddings_sequence = batch['embeddings'].to(self.device)
                sequences = batch['sequences']
                
                # Temporal SAE evaluation
                reconstructed, features = self.temporal_sae(
                    embeddings_sequence,
                    return_features=True
                )
                
                recon_error = F.mse_loss(reconstructed, embeddings_sequence)
                eval_metrics['temporal_recon'] += recon_error.item()
                
                # Feature sparsity
                sparsity = (features.abs() > 0.1).float().mean()
                eval_metrics['feature_sparsity'] += sparsity.item()
                
                # Circuit analysis
                circuit_score = self._evaluate_circuits(features)
                eval_metrics['circuit_coherence'] += circuit_score
                
                # Hierarchical analysis
                hier_score = self._evaluate_hierarchical_mapping(features, sequences)
                eval_metrics['hierarchical_consistency'] += hier_score
                
        # Average metrics
        num_batches = len(val_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        return eval_metrics
        
    def _analyze_circuits(self, features: torch.Tensor):
        """Analyze functional circuits in features."""
        # Convert features to dict format for graph builder
        features_dict = {}
        for layer in range(features.shape[1]):
            features_dict[layer] = features[:, layer].detach()
            
        # Build interaction graph
        graph = self.graph_builder.build_feature_interaction_graph(features_dict)
        
        # Detect motifs
        motifs = self.motif_detector.find_circuit_motifs(graph)
        
        # Validate circuits
        validated_circuits = self.circuit_validator.validate_circuits(
            motifs, features_dict
        )
        
        # Log circuit statistics
        if self.config.get('use_wandb', False):
            wandb.log({
                'num_circuits': len(validated_circuits),
                'avg_circuit_size': np.mean([len(c['instances'][0]) for c in validated_circuits]) if validated_circuits else 0,
                'max_circuit_importance': max([c['importance'] for c in validated_circuits]) if validated_circuits else 0
            }, step=self.global_step)
            
    def _evaluate_circuits(self, features: torch.Tensor) -> float:
        """Evaluate circuit quality."""
        # Simplified circuit coherence metric
        # In practice, this would involve more sophisticated analysis
        
        # Compute feature correlation across layers
        correlations = []
        for layer in range(features.shape[1] - 1):
            curr_feat = features[:, layer].flatten()
            next_feat = features[:, layer + 1].flatten()
            corr = torch.corrcoef(torch.stack([curr_feat, next_feat]))[0, 1]
            correlations.append(corr.item())
            
        return np.mean(correlations) if correlations else 0.0
        
    def _evaluate_hierarchical_mapping(
        self,
        features: torch.Tensor,
        sequences: List[str]
    ) -> float:
        """Evaluate hierarchical feature mapping."""
        # Extract features from last layer for mapping
        last_layer_features = features[:, -1]
        
        # Get predictions from each level
        aa_predictions = self.aa_mapper(last_layer_features, sequences)
        ss_predictions = self.ss_mapper(last_layer_features)
        domain_predictions = self.domain_mapper(last_layer_features)
        
        # Simple consistency metric
        # In practice, this would compare against true annotations
        consistency_scores = []
        
        # Check if hydrophobic residues cluster in predicted domains
        hydro_scores = aa_predictions['hydrophobicity'].squeeze(-1)
        for b in range(len(sequences)):
            if domain_predictions['domains'][b]:
                for domain in domain_predictions['domains'][b]:
                    domain_hydro = hydro_scores[b, domain['start']:domain['end']].mean()
                    consistency_scores.append(domain_hydro.item())
                    
        return np.mean(consistency_scores) if consistency_scores else 0.5
        
    def save_checkpoint(self, path: Path, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'temporal_sae_state': self.temporal_sae.state_dict(),
            'biophysics_sae_state': self.biophysics_sae.state_dict(),
            'aa_mapper_state': self.aa_mapper.state_dict(),
            'ss_mapper_state': self.ss_mapper.state_dict(),
            'domain_mapper_state': self.domain_mapper.state_dict(),
            'temporal_optimizer_state': self.temporal_optimizer.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.temporal_sae.load_state_dict(checkpoint['temporal_sae_state'])
        self.biophysics_sae.load_state_dict(checkpoint['biophysics_sae_state'])
        self.aa_mapper.load_state_dict(checkpoint['aa_mapper_state'])
        self.ss_mapper.load_state_dict(checkpoint['ss_mapper_state'])
        self.domain_mapper.load_state_dict(checkpoint['domain_mapper_state'])
        self.temporal_optimizer.load_state_dict(checkpoint['temporal_optimizer_state'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {path}")


def main():
    """Main training script."""
    # Configuration
    config = {
        'input_dim': 320,  # ESM-2 8M embedding dimension
        'hidden_dim': 512,
        'dict_size': 2560,
        'num_layers': 6,
        'num_attention_heads': 8,
        'dropout': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'sparsity_weight': 0.1,
        'physics_weight': 0.1,
        'batch_size': 32,
        'num_epochs': 100,
        'circuit_analysis_freq': 100,
        'use_wandb': True,
        'wandb_project': 'enhanced-interplm',
        'wandb_entity': 'your-entity'
    }
    
    # Initialize wandb
    if config['use_wandb']:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config=config
        )
        
    # Initialize trainer
    trainer = EnhancedSAETrainer(config)
    
    # Create dummy data loaders (replace with actual data)
    # In practice, you would load ESM embeddings, sequences, structures, etc.
    from torch.utils.data import TensorDataset
    
    # Dummy data
    batch_size = config['batch_size']
    seq_len = 100
    num_samples = 1000
    
    dummy_embeddings = torch.randn(
        num_samples, config['num_layers'], seq_len, config['input_dim']
    )
    dummy_sequences = ['A' * seq_len] * num_samples
    
    train_dataset = TensorDataset(dummy_embeddings[:800])
    val_dataset = TensorDataset(dummy_embeddings[800:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Train
        train_losses = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate
        eval_metrics = trainer.evaluate(val_loader)
        
        # Log metrics
        if config['use_wandb']:
            wandb.log({
                **{f'train_{k}': v for k, v in train_losses.items()},
                **{f'val_{k}': v for k, v in eval_metrics.items()},
                'epoch': epoch
            })
            
        # Save checkpoint if best model
        if eval_metrics['temporal_recon'] < best_val_loss:
            best_val_loss = eval_metrics['temporal_recon']
            trainer.save_checkpoint(
                Path(f'checkpoints/best_model_epoch_{epoch}.pt'),
                epoch
            )
            
        # Update learning rate
        trainer.temporal_scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss = {train_losses['total']:.4f}, "
              f"Val Recon = {eval_metrics['temporal_recon']:.4f}")
              
    if config['use_wandb']:
        wandb.finish()
        

if __name__ == '__main__':
    main()