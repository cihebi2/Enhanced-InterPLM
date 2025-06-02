# enhanced_interplm/examples/train_full_system.py

"""
Complete training example for Enhanced InterPLM framework.
This script demonstrates how to use all components together for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import yaml
import wandb
from tqdm import tqdm
import logging
from datetime import datetime
import os

# Import all Enhanced InterPLM components
from enhanced_interplm.temporal_sae import TemporalSAE, TemporalFeatureTracker
from enhanced_interplm.circuit_discovery import (
    DynamicGraphBuilder, CircuitMotifDetector, CircuitValidator
)
from enhanced_interplm.biophysics import BiophysicsGuidedSAE
from enhanced_interplm.hierarchical_mapping import (
    AminoAcidPropertyMapper, SecondaryStructureMapper,
    DomainFunctionMapper, CrossLevelIntegrator
)
from enhanced_interplm.data_processing import (
    MultiModalProteinDataset, create_protein_dataloaders
)
from enhanced_interplm.evaluation import (
    InterpretabilityMetrics, BiologicalRelevanceScorer
)
from enhanced_interplm.visualization import (
    CircuitVisualizer, TemporalFlowVisualizer,
    HierarchicalMappingVisualizer, BiophysicsConstraintVisualizer
)
from enhanced_interplm.utils.helpers import (
    get_optimal_device, save_checkpoint, load_checkpoint,
    merge_configs, Timer, set_random_seed
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedInterPLMSystem:
    """
    Complete Enhanced InterPLM system integrating all components.
    """
    
    def __init__(self, config_path: str):
        """Initialize the complete system."""
        self.config = self._load_config(config_path)
        self.device = get_optimal_device(self.config.get('gpu_id'))
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
        # Initialize all components
        self._initialize_models()
        self._initialize_optimizers()
        self._initialize_analysis_tools()
        self._initialize_visualization()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize experiment tracking
        if self.config['tracking']['use_wandb']:
            self._init_wandb()
            
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle environment variables in config
        config = self._resolve_env_vars(config)
        
        return config
        
    def _resolve_env_vars(self, config: dict) -> dict:
        """Resolve environment variables in config."""
        import re
        
        def resolve_value(value):
            if isinstance(value, str):
                # Look for ${VAR_NAME} or ${VAR_NAME:default}
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                for match in matches:
                    if ':' in match:
                        var_name, default = match.split(':', 1)
                        var_value = os.environ.get(var_name, default)
                    else:
                        var_value = os.environ.get(match, '')
                    value = value.replace(f'${{{match}}}', var_value)
                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(v) for v in value]
            return value
            
        return resolve_value(config)
        
    def _initialize_models(self):
        """Initialize all model components."""
        model_config = self.config['model']
        
        # Temporal SAE
        self.temporal_sae = TemporalSAE(
            input_dim=model_config['temporal_sae']['input_dim'],
            hidden_dim=model_config['temporal_sae']['hidden_dim'],
            dict_size=model_config['temporal_sae']['dict_size'],
            num_layers=len(model_config['esm_layers']),
            num_attention_heads=model_config['temporal_sae']['num_attention_heads'],
            dropout=model_config['temporal_sae']['dropout'],
            time_decay_factor=model_config['temporal_sae']['time_decay_factor']
        ).to(self.device)
        
        # Feature tracker
        self.feature_tracker = TemporalFeatureTracker(
            feature_dim=model_config['temporal_sae']['dict_size'],
            num_layers=len(model_config['esm_layers'])
        ).to(self.device)
        
        # Biophysics SAE (if enabled)
        if model_config['biophysics']['enabled']:
            self.biophysics_sae = BiophysicsGuidedSAE(
                activation_dim=model_config['temporal_sae']['input_dim'],
                dict_size=model_config['temporal_sae']['dict_size'],
                physics_weight=model_config['biophysics']['physics_weight']
            ).to(self.device)
        else:
            self.biophysics_sae = None
            
        # Hierarchical mappers
        self.hierarchical_mappers = {
            'aa': AminoAcidPropertyMapper(
                feature_dim=model_config['temporal_sae']['dict_size'],
                property_dim=model_config['hierarchical']['aa_property_dim']
            ).to(self.device),
            
            'ss': SecondaryStructureMapper(
                feature_dim=model_config['temporal_sae']['dict_size'],
                hidden_dim=model_config['hierarchical']['ss_hidden_dim']
            ).to(self.device),
            
            'domain': DomainFunctionMapper(
                feature_dim=model_config['temporal_sae']['dict_size'],
                hidden_dim=model_config['hierarchical']['domain_hidden_dim'],
                num_domain_types=model_config['hierarchical']['num_domain_types']
            ).to(self.device)
        }
        
        self.cross_level_integrator = CrossLevelIntegrator().to(self.device)
        
        logger.info("All models initialized successfully")
        
    def _initialize_optimizers(self):
        """Initialize optimizers and schedulers."""
        train_config = self.config['training']
        
        # Collect all parameters
        all_params = []
        all_params.extend(self.temporal_sae.parameters())
        all_params.extend(self.feature_tracker.parameters())
        
        if self.biophysics_sae is not None:
            all_params.extend(self.biophysics_sae.parameters())
            
        for mapper in self.hierarchical_mappers.values():
            all_params.extend(mapper.parameters())
            
        all_params.extend(self.cross_level_integrator.parameters())
        
        # Main optimizer
        if train_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                all_params,
                lr=train_config['learning_rate'],
                weight_decay=train_config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                all_params,
                lr=train_config['learning_rate']
            )
            
        # Learning rate scheduler
        if train_config['scheduler'] == 'cosine_annealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs']
            )
        elif train_config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            self.scheduler = None
            
        logger.info(f"Optimizer: {train_config['optimizer']}, LR: {train_config['learning_rate']}")
        
    def _initialize_analysis_tools(self):
        """Initialize circuit discovery and analysis tools."""
        circuit_config = self.config['model']['circuits']
        
        self.graph_builder = DynamicGraphBuilder(
            mutual_info_threshold=circuit_config['mutual_info_threshold'],
            causal_threshold=circuit_config['causal_threshold']
        )
        
        self.motif_detector = CircuitMotifDetector(
            min_motif_size=circuit_config['min_motif_size'],
            max_motif_size=circuit_config['max_motif_size']
        )
        
        self.circuit_validator = CircuitValidator(
            validation_threshold=circuit_config['validation_threshold']
        )
        
        # Evaluation metrics
        self.metrics_evaluator = InterpretabilityMetrics()
        self.bio_relevance_scorer = BiologicalRelevanceScorer()
        
    def _initialize_visualization(self):
        """Initialize visualization tools."""
        viz_config = self.config['visualization']
        
        self.visualizers = {
            'circuits': CircuitVisualizer(style=viz_config.get('style', 'dark')),
            'temporal': TemporalFlowVisualizer(),
            'hierarchical': HierarchicalMappingVisualizer(),
            'biophysics': BiophysicsConstraintVisualizer()
        }
        
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb.init(
            project=self.config['tracking']['wandb_project'],
            entity=self.config['tracking'].get('wandb_entity'),
            config=self.config,
            name=self.config['experiment']['name']
        )
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Training epoch
            with Timer(f"Training epoch {epoch}", logger):
                train_metrics = self._train_epoch(train_loader)
                
            # Validation
            if epoch % self.config['training']['eval_every'] == 0:
                with Timer(f"Validation epoch {epoch}", logger):
                    val_metrics = self._validate(val_loader)
                    
                # Check for best model
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self._save_best_model()
                    
            # Circuit analysis
            if epoch % self.config['evaluation']['circuit_analysis_freq'] == 0:
                with Timer("Circuit analysis", logger):
                    self._analyze_circuits(val_loader)
                    
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
                    
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self._save_checkpoint(epoch)
                
        logger.info("Training completed!")
        
    def _train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.temporal_sae.train()
        if self.biophysics_sae is not None:
            self.biophysics_sae.train()
        for mapper in self.hierarchical_mappers.values():
            mapper.train()
            
        epoch_losses = {
            'reconstruction': 0,
            'sparsity': 0,
            'physics': 0,
            'circuit_coherence': 0,
            'hierarchical': 0,
            'total': 0
        }
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            embeddings = batch['embeddings'].to(self.device)
            sequences = batch['sequences']
            lengths = batch['lengths'].to(self.device)
            
            # Optional data
            structures = batch.get('structures', None)
            if structures is not None:
                structures = structures.to(self.device)
                
            annotations = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k.startswith(('secondary_structure', 'domains', 'go_terms'))
            }
            
            # Forward pass through temporal SAE
            reconstructed, features = self.temporal_sae(embeddings, return_features=True)
            
            # Compute losses
            losses = self._compute_losses(
                embeddings, reconstructed, features,
                sequences, structures, annotations, lengths
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.temporal_sae.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            self.global_step += 1
            
            # Update epoch losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
                
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total'].item(),
                'recon': losses['reconstruction'].item()
            })
            
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
        
    def _compute_losses(
        self,
        embeddings: torch.Tensor,
        reconstructed: torch.Tensor,
        features: torch.Tensor,
        sequences: list,
        structures: torch.Tensor,
        annotations: dict,
        lengths: torch.Tensor
    ) -> dict:
        """Compute all loss components."""
        loss_weights = self.config['training']['losses']
        losses = {}
        
        # Reconstruction loss
        recon_loss = self._masked_mse_loss(reconstructed, embeddings, lengths)
        losses['reconstruction'] = recon_loss * loss_weights['reconstruction']
        
        # Sparsity loss
        sparsity_loss = features.abs().mean()
        losses['sparsity'] = sparsity_loss * loss_weights['sparsity']
        
        # Biophysics loss (if enabled)
        if self.biophysics_sae is not None and structures is not None:
            last_layer_embeds = embeddings[:, -1]
            bio_recon, bio_features, physics_losses = self.biophysics_sae(
                last_layer_embeds,
                sequences=sequences,
                structures=structures,
                return_physics_loss=True
            )
            
            physics_loss = sum(physics_losses.values()) if physics_losses else 0
            losses['physics'] = physics_loss * loss_weights['physics']
        else:
            losses['physics'] = torch.tensor(0.0).to(self.device)
            
        # Circuit coherence loss
        circuit_loss = self._compute_circuit_coherence_loss(features)
        losses['circuit_coherence'] = circuit_loss * loss_weights['circuit_coherence']
        
        # Hierarchical consistency loss
        hier_loss = self._compute_hierarchical_loss(features, sequences, annotations)
        losses['hierarchical'] = hier_loss * loss_weights['hierarchical_consistency']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
        
    def _masked_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute MSE loss with masking for variable length sequences."""
        batch_size, num_layers, max_len, _ = pred.shape
        
        # Create mask
        mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).expand(batch_size, num_layers, max_len).to(pred.device)
        
        # Apply mask and compute loss
        masked_pred = pred[mask]
        masked_target = target[mask]
        
        return nn.functional.mse_loss(masked_pred, masked_target)
        
    def _compute_circuit_coherence_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Compute loss to encourage coherent circuit activation."""
        # Compute feature correlation across adjacent layers
        correlations = []
        
        for layer in range(features.shape[1] - 1):
            curr_feat = features[:, layer]
            next_feat = features[:, layer + 1]
            
            # Normalize features
            curr_norm = nn.functional.normalize(curr_feat, p=2, dim=-1)
            next_norm = nn.functional.normalize(next_feat, p=2, dim=-1)
            
            # Compute correlation
            corr = torch.bmm(curr_norm, next_norm.transpose(1, 2))
            
            # Encourage sparse but strong connections
            coherence = -torch.mean(corr.max(dim=-1)[0])  # Maximize strongest connections
            correlations.append(coherence)
            
        return torch.mean(torch.stack(correlations))
        
    def _compute_hierarchical_loss(
        self,
        features: torch.Tensor,
        sequences: list,
        annotations: dict
    ) -> torch.Tensor:
        """Compute hierarchical mapping consistency loss."""
        last_layer_features = features[:, -1]
        
        total_loss = torch.tensor(0.0).to(self.device)
        
        # AA property prediction loss
        if 'hydrophobicity' in annotations:
            aa_props = self.hierarchical_mappers['aa'](last_layer_features, sequences)
            # Simplified: just check hydrophobicity prediction
            hydro_pred = aa_props['hydrophobicity']
            hydro_target = annotations.get('hydrophobicity', hydro_pred)  # Use pred as target if not available
            total_loss += nn.functional.mse_loss(hydro_pred, hydro_target)
            
        # Secondary structure prediction loss
        if 'secondary_structure' in annotations:
            ss_pred = self.hierarchical_mappers['ss'](last_layer_features)
            ss_target = annotations['secondary_structure']
            total_loss += nn.functional.cross_entropy(
                ss_pred['ss_probabilities'].reshape(-1, 8),
                ss_target.reshape(-1).long()
            )
            
        return total_loss
        
    def _validate(self, val_loader: DataLoader) -> dict:
        """Validate the model."""
        self.temporal_sae.eval()
        if self.biophysics_sae is not None:
            self.biophysics_sae.eval()
        for mapper in self.hierarchical_mappers.values():
            mapper.eval()
            
        val_metrics = {
            'reconstruction': 0,
            'sparsity': 0,
            'variance_explained': 0,
            'feature_entropy': 0,
            'circuit_coherence': 0,
            'total_loss': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                embeddings = batch['embeddings'].to(self.device)
                sequences = batch['sequences']
                lengths = batch['lengths'].to(self.device)
                
                # Forward pass
                reconstructed, features = self.temporal_sae(embeddings, return_features=True)
                
                # Compute metrics
                metrics = self.metrics_evaluator.compute_all_metrics(
                    features[:, -1],  # Use last layer features
                    reconstructed[:, -1],
                    embeddings[:, -1]
                )
                
                # Update validation metrics
                val_metrics['reconstruction'] += metrics['reconstruction']['mse'].value
                val_metrics['sparsity'] += metrics['features']['l0_sparsity'].value
                val_metrics['variance_explained'] += metrics['reconstruction']['variance_explained'].value
                val_metrics['feature_entropy'] += metrics['features']['feature_entropy'].value
                
                # Compute total loss for model selection
                losses = self._compute_losses(
                    embeddings, reconstructed, features,
                    sequences, None, {}, lengths
                )
                val_metrics['total_loss'] += losses['total'].item()
                
        # Average metrics
        num_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
            
        return val_metrics
        
    def _analyze_circuits(self, data_loader: DataLoader):
        """Perform detailed circuit analysis."""
        all_features = []
        
        # Collect features from multiple batches
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 10:  # Limit to 10 batches for efficiency
                    break
                    
                embeddings = batch['embeddings'].to(self.device)
                _, features = self.temporal_sae(embeddings, return_features=True)
                all_features.append(features)
                
        # Concatenate features
        all_features = torch.cat(all_features, dim=0)
        
        # Convert to dict format for graph builder
        features_dict = {}
        for layer in range(all_features.shape[1]):
            features_dict[layer] = all_features[:, layer].detach()
            
        # Build interaction graph
        interaction_graph = self.graph_builder.build_feature_interaction_graph(features_dict)
        
        # Detect motifs
        motifs = self.motif_detector.find_circuit_motifs(
            interaction_graph,
            min_frequency=2
        )
        
        # Validate circuits
        validated_circuits = self.circuit_validator.validate_circuits(
            motifs[:self.config['evaluation']['max_circuits_to_analyze']],
            features_dict
        )
        
        # Visualize top circuits
        if self.config['visualization']['plot_circuits']:
            output_dir = Path(self.config['tracking']['results_dir']) / 'circuits'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, circuit in enumerate(validated_circuits[:5]):
                fig = self.visualizers['circuits'].visualize_circuit_graph(
                    circuit,
                    interaction_graph,
                    save_path=output_dir / f'circuit_{self.current_epoch}_{i}.html'
                )
                
        # Log circuit statistics
        circuit_stats = {
            'num_circuits': len(validated_circuits),
            'avg_circuit_size': np.mean([len(c['instances'][0]) for c in validated_circuits]) if validated_circuits else 0,
            'max_importance': max([c['importance'] for c in validated_circuits]) if validated_circuits else 0,
            'avg_coherence': np.mean([c['validation_scores']['coherence'] for c in validated_circuits]) if validated_circuits else 0
        }
        
        logger.info(f"Circuit analysis: {circuit_stats}")
        
        if self.config['tracking']['use_wandb']:
            wandb.log(circuit_stats, step=self.global_step)
            
    def _log_metrics(self, train_metrics: dict, val_metrics: dict):
        """Log metrics to tracking service."""
        all_metrics = {
            **{f'train/{k}': v for k, v in train_metrics.items()},
            **{f'val/{k}': v for k, v in val_metrics.items()},
            'epoch': self.current_epoch,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if self.config['tracking']['use_wandb']:
            wandb.log(all_metrics, step=self.global_step)
            
        # Also log to console
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss={train_metrics['total']:.4f}, "
            f"Val Loss={val_metrics['total_loss']:.4f}, "
            f"Val VarExp={val_metrics['variance_explained']:.4f}"
        )
        
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['tracking']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'config': self.config,
            'model_states': {
                'temporal_sae': self.temporal_sae.state_dict(),
                'feature_tracker': self.feature_tracker.state_dict(),
                'aa_mapper': self.hierarchical_mappers['aa'].state_dict(),
                'ss_mapper': self.hierarchical_mappers['ss'].state_dict(),
                'domain_mapper': self.hierarchical_mappers['domain'].state_dict(),
                'cross_level_integrator': self.cross_level_integrator.state_dict()
            },
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None
        }
        
        if self.biophysics_sae is not None:
            checkpoint['model_states']['biophysics_sae'] = self.biophysics_sae.state_dict()
            
        save_checkpoint(checkpoint, checkpoint_path)
        
        # Keep only the most recent checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)
        
    def _save_best_model(self):
        """Save the best model."""
        best_model_path = Path(self.config['tracking']['checkpoint_dir']) / 'best_model.pt'
        self._save_checkpoint('best')
        
        # Also save a copy with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_model_copy = Path(self.config['tracking']['checkpoint_dir']) / f'best_model_{timestamp}.pt'
        
        import shutil
        shutil.copy(best_model_path, best_model_copy)
        
        logger.info(f"Saved best model with val_loss={self.best_val_loss:.4f}")
        
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Keep only the most recent k checkpoints."""
        keep_best_k = self.config['training']['keep_best_k']
        
        # List all checkpoint files
        checkpoint_files = sorted(
            checkpoint_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Delete old checkpoints
        for checkpoint_file in checkpoint_files[keep_best_k:]:
            checkpoint_file.unlink()
            

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Enhanced InterPLM')
    parser.add_argument(
        '--config',
        type=str,
        default='enhanced_interplm/configs/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Override data root directory'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Override experiment name'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = EnhancedInterPLMSystem(args.config)
    
    # Override config if specified
    if args.data_root:
        system.config['data']['root_dir'] = args.data_root
    if args.experiment_name:
        system.config['experiment']['name'] = args.experiment_name
        
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = load_checkpoint(args.resume, system.device)
        system.temporal_sae.load_state_dict(checkpoint['model_states']['temporal_sae'])
        system.optimizer.load_state_dict(checkpoint['optimizer_state'])
        system.current_epoch = checkpoint['epoch'] + 1
        system.global_step = checkpoint['global_step']
        logger.info(f"Resumed from checkpoint: {args.resume}")
        
    # Create data loaders
    train_loader, val_loader, test_loader = create_protein_dataloaders(
        data_root=Path(system.config['data']['root_dir']),
        batch_size=system.config['data']['batch_size'],
        num_workers=system.config['data']['num_workers'],
        layers=system.config['model']['esm_layers'],
        include_structures=system.config['data']['include_structures'],
        include_evolution=system.config['data']['include_evolution'],
        include_annotations=system.config['data']['include_annotations']
    )
    
    # Train the system
    system.train(train_loader, val_loader)
    
    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = system._validate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Save final results
    results_dir = Path(system.config['tracking']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'config': system.config,
            'best_val_loss': system.best_val_loss
        }, f, indent=2)
        
    if system.config['tracking']['use_wandb']:
        wandb.finish()
        

if __name__ == '__main__':
    main()