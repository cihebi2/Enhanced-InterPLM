# Enhanced InterPLM: 蛋白质语言模型的高级可解释性框架

[English](https://claude.ai/chat/README_en.md) | 中文

## 🚀 项目概述

Enhanced InterPLM 是一个创新的框架，通过先进的稀疏自编码器技术来理解和解释蛋白质语言模型（PLMs）。该项目在原始 InterPLM 的基础上，引入了四大核心创新：

### 🔬 核心创新

1. **时序感知稀疏自编码器 (Temporal-SAE)**
   * 捕获特征在Transformer层间的演化
   * 使用LSTM编码时序依赖关系
   * 多头注意力机制处理跨层特征交互
2. **功能回路自动发现**
   * 动态构建特征交互图
   * 检测特征激活模式中的基序
   * 因果推断理解特征关系
3. **生物物理约束学习**
   * 整合疏水性、电荷和大小约束
   * 氢键网络建模
   * 空间相互作用预测
4. **层次化特征-功能映射**
   * 氨基酸属性预测
   * 二级结构分类
   * 功能域边界检测

## 📋 目录

* [安装](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E5%AE%89%E8%A3%85)
* [快速开始](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
* [主要功能](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E4%B8%BB%E8%A6%81%E5%8A%9F%E8%83%BD)
* [项目结构](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84)
* [使用示例](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E4%BD%BF%E7%94%A8%E7%A4%BA%E4%BE%8B)
* [评估指标](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87)
* [引用](https://claude.ai/chat/2e98c881-df72-4e9e-a2dd-35ee3a74a7c2#%E5%BC%95%E7%94%A8)

## 🛠 安装

### 使用 Conda（推荐）

```bash
# 克隆仓库
git clone https://github.com/your-username/enhanced-interplm.git
cd enhanced-interplm

# 创建 conda 环境
conda env create -f environment.yml
conda activate enhanced-interplm

# 安装包
pip install -e .
```

### 使用 pip

```bash
# 克隆仓库
git clone https://github.com/your-username/enhanced-interplm.git
cd enhanced-interplm

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 🚀 快速开始

### 1. 提取 ESM 嵌入

```python
from enhanced_interplm.esm.layerwise_embeddings import extract_multilayer_embeddings

# 从多个层提取嵌入
embeddings = extract_multilayer_embeddings(
    sequences=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
    model_name="esm2_t6_8M_UR50D",
    layers=[1, 2, 3, 4, 5, 6]
)
```

### 2. 训练 Temporal-SAE

```python
from enhanced_interplm.temporal_sae import TemporalSAE
from enhanced_interplm.train_enhanced_sae import EnhancedSAETrainer

# 配置训练
config = {
    'input_dim': 320,
    'hidden_dim': 512,
    'dict_size': 2560,
    'num_layers': 6,
    'learning_rate': 1e-3,
    'num_epochs': 100
}

# 初始化并训练
trainer = EnhancedSAETrainer(config)
trainer.train_epoch(train_loader, epoch=0)
```

### 3. 发现功能回路

```python
from enhanced_interplm.circuit_discovery import DynamicGraphBuilder, CircuitMotifDetector

# 构建特征交互图
graph_builder = DynamicGraphBuilder()
interaction_graph = graph_builder.build_feature_interaction_graph(features)

# 检测回路基序
motif_detector = CircuitMotifDetector()
circuits = motif_detector.find_circuit_motifs(interaction_graph)
```

## 🎯 主要功能

### 时序特征追踪

跟踪特征如何在Transformer层间演化：

```python
from enhanced_interplm.esm.feature_evolution import FeatureEvolutionAnalyzer

analyzer = FeatureEvolutionAnalyzer(num_layers=6, feature_dim=2560)
analyzer.track_batch(embeddings)
evolution_metrics = analyzer.analyze_evolution()

# 可视化演化模式
fig = analyzer.visualize_evolution(evolution_metrics, save_path="evolution.png")
```

### 生物物理约束

应用物理化学约束来指导特征学习：

```python
from enhanced_interplm.biophysics import BiophysicsGuidedSAE

bio_sae = BiophysicsGuidedSAE(
    activation_dim=320,
    dict_size=2560,
    physics_weight=0.1
)

# 使用物理约束训练
reconstructed, features, physics_losses = bio_sae(
    embeddings,
    sequences=sequences,
    structures=structures,
    return_physics_loss=True
)
```

### 层次化分析

多尺度特征到功能的映射：

```python
from enhanced_interplm.hierarchical_mapping import (
    AminoAcidPropertyMapper,
    SecondaryStructureMapper,
    DomainFunctionMapper,
    CrossLevelIntegrator
)

# 初始化映射器
aa_mapper = AminoAcidPropertyMapper(feature_dim=2560)
ss_mapper = SecondaryStructureMapper(feature_dim=2560)
domain_mapper = DomainFunctionMapper(feature_dim=2560)

# 获取多尺度预测
aa_properties = aa_mapper(features, sequences)
ss_predictions = ss_mapper(features)
domains = domain_mapper(features)

# 跨层级整合
integrator = CrossLevelIntegrator()
integrated_features = integrator(aa_properties, ss_predictions, domains)
```

## 📁 项目结构

```
enhanced-interplm/
├── temporal_sae/              # 时序感知SAE实现
│   ├── temporal_autoencoder.py
│   ├── attention_modules.py
│   └── layer_tracking.py
├── circuit_discovery/         # 回路检测算法
│   ├── graph_builder.py
│   ├── motif_detector.py
│   └── causal_inference.py
├── biophysics/               # 生物物理约束
│   ├── physics_constraints.py
│   ├── hydrophobic_module.py
│   └── electrostatic_module.py
├── hierarchical_mapping/     # 多尺度特征映射
│   ├── aa_property_mapper.py
│   ├── secondary_structure.py
│   └── domain_function.py
├── evaluation/               # 评估指标
│   ├── comprehensive_metrics.py
│   └── biological_relevance.py
├── visualization/            # 可视化工具
│   ├── circuit_visualizer.py
│   └── temporal_flow.py
└── examples/                 # 使用示例
    └── train_full_system.py
```

## 📊 评估指标

Enhanced InterPLM 提供全面的评估指标：

### 可解释性指标

* **重建质量** : 输入和重建嵌入之间的MSE
* **特征稀疏性** : 每个位置的活跃特征百分比
* **回路连贯性** : 回路激活模式的一致性
* **方差解释** : 模型解释的数据方差比例

### 生物学相关性

* **二级结构预测准确率**
* **功能域检测 AUC**
* **GO术语预测 F1分数**
* **进化保守性相关性**

### 物理合规性

* **疏水性聚类质量**
* **静电相互作用准确性**
* **氢键网络保真度**

## 💻 命令行工具

### 提取嵌入

```bash
enhanced-interplm-extract input.fasta output.h5 \
    --model esm2_t6_8M_UR50D \
    --layers 1,2,3,4,5,6 \
    --batch-size 8
```

### 分析特征

```bash
enhanced-interplm-analyze checkpoint.pt data.h5 \
    --output analysis_results/ \
    --top-features 100 \
    --circuits
```

## 🔬 高级用法

### 自定义回路分析

```python
# 定义自定义回路验证标准
validator = CircuitValidator(validation_threshold=0.8)
validated_circuits = validator.validate_circuits(
    circuits,
    features,
    task_labels=functional_annotations
)

# 可视化顶级回路
from enhanced_interplm.visualization import CircuitVisualizer
visualizer = CircuitVisualizer()
visualizer.plot_circuit_activation(validated_circuits[0], features)
```

### 批量处理

```python
# 处理大规模数据集
from enhanced_interplm.data_processing import create_protein_dataloaders

train_loader, val_loader, test_loader = create_protein_dataloaders(
    data_root=Path("data/processed"),
    batch_size=32,
    num_workers=4,
    include_structures=True,
    include_evolution=True
)
```

## 📈 训练细节

### 数据要求

1. **蛋白质序列** : FASTA格式
2. **ESM嵌入** : 预先提取或实时计算
3. **结构数据** （可选）: PDB文件或AlphaFold预测
4. **功能注释** （可选）: UniProt注释、GO术语

### 关键超参数

* `dict_size`: SAE中的特征数量（通常是输入维度的8-16倍）
* `num_layers`: 要分析的transformer层数
* `physics_weight`: 生物物理约束的强度（0.05-0.2）
* `sparsity_weight`: L1正则化强度（0.05-0.2）

### 计算要求

* **GPU** : 建议使用16GB+ VRAM的NVIDIA GPU
* **内存** : 大型蛋白质数据集需要32GB+ RAM
* **存储** : 预处理嵌入约需100GB

## 📚 引用

如果您在研究中使用了 Enhanced InterPLM，请引用：

```bibtex
@article{enhanced-interplm2024,
  title={Enhanced InterPLM: 蛋白质语言模型的多尺度可解释性},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## 🙏 致谢

本项目基于 Simon & Zou (2024) 的原始 InterPLM 框架，并融合了机制可解释性和蛋白质生物物理学的最新进展。

## 📄 许可证

MIT 许可证 - 详见 LICENSE 文件

## 🤝 贡献

欢迎贡献！请查看 CONTRIBUTING.md 了解指南。

## 📧 联系方式

如有问题或合作意向，请提交 issue 或联系：your-email@institution.edu

---

## 🎯 路线图

### 第一阶段：基础设施（已完成）

* ✅ 时序SAE实现
* ✅ 多层嵌入提取
* ✅ 基础评估指标

### 第二阶段：核心创新（进行中）

* ✅ 功能回路发现
* ✅ 生物物理约束
* 🔄 层次化映射优化
* 🔄 大规模验证

### 第三阶段：应用（计划中）

* ⏳ 功能预测工具
* ⏳ 蛋白质设计指导
* ⏳ 疾病相关特征分析

## 💡 常见问题

**Q: Enhanced InterPLM 与原始 InterPLM 的主要区别是什么？**

A: Enhanced InterPLM 引入了四大创新：(1) 时序特征追踪，(2) 功能回路自动发现，(3) 生物物理约束，(4) 层次化映射。这些创新使得模型能够更好地理解PLM的内部工作机制。

**Q: 需要多少GPU内存？**

A: 对于ESM-2 8M模型，建议至少16GB GPU内存。对于更大的模型（如650M），建议使用多GPU设置。

**Q: 可以用于哪些下游任务？**

A: Enhanced InterPLM 可用于：蛋白质功能预测、结构-功能关系分析、进化分析、蛋白质设计指导等。
