
# Enhanced InterPLM: Advanced Interpretability for Protein Language Models

## Overview

Enhanced InterPLM is an advanced framework for understanding and interpreting protein language models (PLMs) through innovative sparse autoencoder techniques. This project extends the original InterPLM with four key innovations:

1. **Temporal-SAE** : Captures feature evolution across transformer layers
2. **Functional Circuit Discovery** : Automatically identifies feature interaction patterns
3. **Biophysics-Guided Learning** : Incorporates physical constraints into feature learning
4. **Hierarchical Feature-Function Mapping** : Multi-scale analysis from amino acids to domains

## Key Features

### 1. Temporal-Aware Sparse Autoencoders

* LSTM-based encoding to capture temporal dependencies across layers
* Multi-head attention for cross-layer feature interactions
* Time decay factors for modeling direct and indirect influences

### 2. Automatic Circuit Discovery

* Dynamic graph construction from feature interactions
* Motif detection in feature activation patterns
* Causal inference for understanding feature relationships
* Circuit validation through coherence and stability metrics

### 3. Biophysical Constraints

* Integration of hydrophobicity, charge, and size constraints
* Hydrogen bond network modeling
* Spatial interaction predictions
* Physics-guided feature regularization

### 4. Hierarchical Mapping

* Amino acid property prediction
* Secondary structure classification
* Domain boundary detection and classification
* Cross-level feature integration

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/enhanced-interplm.git
cd enhanced-interplm

# Create conda environment
conda env create -f environment.yml
conda activate enhanced-interplm

# Install the package
pip install -e .
```

## Quick Start

### 1. Extract Multi-Layer ESM Embeddings

```python
from enhanced_interplm.esm.layerwise_embeddings import extract_multilayer_embeddings

# Extract embeddings from all layers
embeddings = extract_multilayer_embeddings(
    sequences=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
    model_name="esm2_t6_8M_UR50D",
    layers=list(range(1, 7))
)
```

### 2. Train Temporal-SAE

```python
from enhanced_interplm.temporal_sae import TemporalSAE
from enhanced_interplm.train_enhanced_sae import EnhancedSAETrainer

# Configure training
config = {
    'input_dim': 320,
    'hidden_dim': 512,
    'dict_size': 2560,
    'num_layers': 6,
    'learning_rate': 1e-3,
    'num_epochs': 100
}

# Initialize and train
trainer = EnhancedSAETrainer(config)
trainer.train_epoch(train_loader, epoch=0)
```

### 3. Discover Functional Circuits

```python
from enhanced_interplm.circuit_discovery import DynamicGraphBuilder, CircuitMotifDetector

# Build feature interaction graph
graph_builder = DynamicGraphBuilder()
interaction_graph = graph_builder.build_feature_interaction_graph(features)

# Detect circuit motifs
motif_detector = CircuitMotifDetector()
circuits = motif_detector.find_circuit_motifs(interaction_graph)
```

### 4. Apply Biophysical Constraints

```python
from enhanced_interplm.biophysics import BiophysicsGuidedSAE

# Create biophysics-aware SAE
bio_sae = BiophysicsGuidedSAE(
    activation_dim=320,
    dict_size=2560,
    physics_weight=0.1
)

# Train with physical constraints
reconstructed, features, physics_losses = bio_sae(
    embeddings,
    sequences=sequences,
    structures=structures,
    return_physics_loss=True
)
```

## Project Structure

```
enhanced-interplm/
├── temporal_sae/              # Temporal-aware SAE implementation
│   ├── temporal_autoencoder.py
│   ├── attention_modules.py
│   └── layer_tracking.py
├── circuit_discovery/         # Circuit detection algorithms
│   ├── graph_builder.py
│   ├── motif_detector.py
│   └── causal_inference.py
├── biophysics/               # Biophysical constraints
│   ├── physics_constraints.py
│   ├── hydrophobic_module.py
│   └── electrostatic_module.py
├── hierarchical_mapping/     # Multi-scale feature mapping
│   ├── aa_property_mapper.py
│   ├── secondary_structure.py
│   └── domain_function.py
└── train_enhanced_sae.py     # Main training script
```

## Advanced Usage

### Custom Circuit Analysis

```python
# Define custom circuit validation criteria
validator = CircuitValidator(validation_threshold=0.8)
validated_circuits = validator.validate_circuits(
    circuits,
    features,
    task_labels=functional_annotations
)

# Visualize top circuits
from enhanced_interplm.visualization import CircuitVisualizer
visualizer = CircuitVisualizer()
visualizer.plot_circuit_activation(validated_circuits[0], features)
```

### Hierarchical Feature Analysis

```python
from enhanced_interplm.hierarchical_mapping import (
    AminoAcidPropertyMapper,
    SecondaryStructureMapper,
    DomainFunctionMapper,
    CrossLevelIntegrator
)

# Map features to different levels
aa_mapper = AminoAcidPropertyMapper(feature_dim=2560)
ss_mapper = SecondaryStructureMapper(feature_dim=2560)
domain_mapper = DomainFunctionMapper(feature_dim=2560)

# Get multi-scale predictions
aa_properties = aa_mapper(features, sequences)
ss_predictions = ss_mapper(features)
domains = domain_mapper(features)

# Integrate across levels
integrator = CrossLevelIntegrator()
integrated_features = integrator(aa_properties, ss_predictions, domains)
```

## Training Details

### Data Requirements

1. **Protein Sequences** : FASTA format
2. **ESM Embeddings** : Pre-extracted or computed on-the-fly
3. **Structural Data** (optional): PDB files or AlphaFold predictions
4. **Functional Annotations** (optional): UniProt annotations, GO terms

### Hyperparameters

Key hyperparameters to tune:

* `dict_size`: Number of features in SAE (typically 8-16x input dimension)
* `num_layers`: Number of transformer layers to analyze
* `physics_weight`: Strength of biophysical constraints (0.05-0.2)
* `sparsity_weight`: L1 regularization strength (0.05-0.2)

### Computational Requirements

* **GPU** : NVIDIA GPU with 16GB+ VRAM recommended
* **Memory** : 32GB+ RAM for large protein datasets
* **Storage** : ~100GB for preprocessed embeddings

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

1. **Reconstruction Quality** : MSE between input and reconstructed embeddings
2. **Feature Sparsity** : Percentage of active features per position
3. **Circuit Coherence** : Consistency of circuit activation patterns
4. **Biological Relevance** : Correlation with known functional annotations
5. **Physics Compliance** : Adherence to biophysical constraints

## Citation

If you use Enhanced InterPLM in your research, please cite:

```bibtex
@article{enhanced-interplm2024,
  title={Enhanced InterPLM: Multi-Scale Interpretability for Protein Language Models},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## Acknowledgments

This project builds upon the original InterPLM framework by Simon & Zou (2024) and incorporates ideas from recent advances in mechanistic interpretability and protein biophysics.

## License

MIT License - see LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions or collaborations, please open an issue or contact: your-email@institution.edu
