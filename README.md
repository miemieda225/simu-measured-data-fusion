# Simu-Measured Data Fusion: 跨域数据融合与物理意义演化分析

本项目为硕士论文核心代码库，面向道面结构性能演化的数据分析与力学仿真融合方法。

### 算法创新与核心逻辑
- **解耦表示与物理演化分析：** 采用 **$\beta$-VAE** 对高维结构数据进行稀疏降维。通过对潜在空间 (Latent Space) 进行插值与遍历扰动，实现对结构物理特性的解耦分析与机理定量解释。
- **双流特征融合架构：** 设计 **双 Encoder 单 Decoder** 网络，引入 **注意力机制 (Attention Mechanism)** 实现多源异构数据的特征级融合。

### 核心模块说明
- **`Network/`**: 针对不同来源数据的 CNN-VAE 架构：
  - **单板应变数据:** $41 \times 41$ 空间特征提取。
  - **互功率谱 (CPSD):** $513 \times 125$ 频域特征提取。
- **`Util/`**: 自动化数据预处理流水线（Data Pipeline）、评估指标、物理信息损失函数。

### 核心文件说明
1. **特征压缩:** `train_NN_41` / `measured_data_train` (实测数据压缩表示)。
2. **物理意义分析:** `raodong.ipynb` —— 潜在空间流形分析。
3. **跨域生成:** `measured_data_simulate` —— 基于生成模型的数据扩增。
4. **决策融合:** `fusion_v1.ipynb` —— 多源数据集成验证。

### 🛠 技术栈
- **框架:** PyTorch, NumPy, Matplotlib.
- **算法:** $\beta$-VAE, CNN, Attention Mechanism, Feature Fusion.
- **工具:** Git 版本管理, 模块化 Python 开发。
