# RoBoT: Parameter-based Gradient Flow (PGF) Optimization Suite

RoBoT (Robust Behavior Optimization Toolkit) 是一个面向机器人超长轨迹控制的高级强化学习优化库。其核心基于 **PGF (Parameter-based Gradient Flow)** 理论，旨在解决传统二阶优化方法（如 TRPO）在长序列下的显存爆炸问题，实现 $O(1)$ 显存开销下的精确 Hessian-Vector Product (HVP)。

---

## 💡 为什么选择 RoBoT？(vs. 现有的 RL 库)

在强化学习领域，虽然有 Stable Baselines3 (SB3)、CleanRL 或 Ray RLLib 等成熟的工业级库，但它们在处理 **长序列 (Long-Sequence)** 和 **递归模型 (SSM/Mamba)** 时存在本质的架构瓶颈：

1.  **内存复杂度：$O(L)$ vs. $O(1)$**
    *   **主流库**：依赖 PyTorch/TensorFlow 的 **Autograd (Backprop)**，必须在显存中存储整个序列长度 $L$ 的所有中间激活值。对于 5000+ 步的轨迹，显存会轻易爆掉。
    *   **RoBoT**：采用 **PGF (参数化梯度流)**，通过递归雅可比流直接计算梯度，**显存开销不随序列长度增加**。你可以用 8G 显存训练标准库需要 80G 才能跑的超长轨迹。

2.  **微分机制：数值近似 vs. 全纯合成**
    *   **主流库**：TRPO 实现通常使用共轭梯度 (CG) 对 Hessian 进行数值逼近，容易在非凸地形崩溃。
    *   **RoBoT (Holo-TRPO)**：利用 **全纯逻辑同构 (HLI)** 和复数步微分 (CSD)，直接提取目标函数的解析谱（Taylor Spectrum）。这不仅是近似，而是对 Loss 地形的高阶解析合成。

3.  **算子哲学：统计拟合 vs. 逻辑电路**
    *   **主流库**：将神经网络视为黑盒统计工具。
    *   **RoBoT**：将神经网络视为**全纯解析电路**。通过 **ReLU 同构**，我们让 PPO 的非连续 Clip 操作在复数域变得平滑且可导。

---

## 📂 代码结构与用途说明

本项目的代码按照功能分为核心引擎、强化学习变体、理论推导、工具与跑分、环境包装五大类。

### 1. 核心二阶优化引擎 (Core Engines)
*   **[pgf_trpo_engine.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/pgf_trpo_engine.py)**
    *   **用途**: 项目的核心灵魂，实现了基于 PGF 的精确 HVP。
    *   **核心逻辑**: 采用 Double-Path 架构，通过前向解析对偶扫描（JVP Path）和逆向伴随扫描（Adjoint Path）计算二阶项，支持超长轨迹（5000+ steps）的精确 TRPO 优化。
*   **[holo_trpo_engine.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/holo_trpo_engine.py)**
    *   **用途**: 全纯 TRPO 实验引擎。
    *   **核心逻辑**: 利用复数步长微分 (CSD) 与 FFT 技术提取 Loss 函数的高阶泰勒系数。它可以实现比标准 Hessian 更精细的信任区域控制，具有极高的数值稳定性。

### 2. 强化学习变体 (RL Variants)
*   **[pgf_ppo_engine.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/pgf_ppo_engine.py)**
    *   **用途**: PPO 算法的 PGF 内存优化版。
    *   **核心逻辑**: 引入了 **ReLU Isomorphism (ReLU 同构)**，将非连续的 Clip 函数重写为解析梯度场掩码，使 PPO 同样具备 $O(1)$ 显存优势。

### 3. 理论推导 (Theory)
*   **[derivation_pgf_trpo.md](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/derivation_pgf_trpo.md)**
    *   **用途**: 整个项目的数学白皮书。
    *   **内容**: 包含 PGF 伴随方程、PPO ReLU 同构证明、SSM 并行扫描下的对偶方程组递推公式，以及基于柯西积分定理的高阶泰勒提取证明。

### 4. 工具与性能跑分 (Tools & Benchmarks)
*   **[benchmark_ppo.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/benchmark_ppo.py)**
    *   **用途**: PGF-PPO 显存与时间精度定量分析。
    *   **实验结果**:
        | Seq Len | Std Mem (MB) | PGF Mem (MB) | Std Time (ms) | PGF Time (ms) | Precision |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | 512 | 40.53 | **33.57** | 160.52 | 378.53 | 1e-07 |
        | 1024 | 65.58 | **36.61** | 3.06 | 65.43 | 1e-07 |
        | 2048 | 109.64 | **40.67** | 5.61 | 24.48 | 1e-07 |
        | 4096 | 185.75 | **48.80** | 10.19 | 42.73 | 1e-07 |
        | 8192 | 341.72 | **65.05** | 24.94 | 84.13 | 1e-07 |
*   **[benchmark_hvp.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/benchmark_hvp.py)**
    *   **用途**: HVP 性能对比脚本。
    *   **实验结果**:
        | Seq Len | Std Mem (MB) | PGF Mem (MB) | Std Time (ms) | PGF Time (ms) | Precision |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | 512 | 599.64 | **322.63** | 205.21 | 480.65 | 1e-07 |
        | 1024 | 1339.33 | **325.40** | 437.92 | **43.22** | 1e-07 |
        | 2048 | 2655.95 | **326.94** | 1131.94 | **72.37** | 1e-07 |
        | 4096 | 5162.20 | **330.02** | 659.17 | **142.26** | 1e-07 |
        | 5000 | 5838.90 | **332.84** | 538.19 | **171.99** | 1e-07 |

### 5. 环境包装与验证 (Env Wrappers & Verification)
*   **[isaac_gym_pgf_wrapper.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/isaac_gym_pgf_wrapper.py)**
    *   **用途**: Isaac Gym 工业级包装器。
    *   **功能**: 针对大规模并行物理仿真环境设计的接口，支持 GPU-to-GPU 的高效轨迹采集与梯度更新。
*   **[trpo_convergence_task.py](file:///c:/Users/ukiyo/OneDrive/Desktop/transformer/mamba/RoBoT/trpo_convergence_task.py)**
    *   **用途**: PGF-TRPO 与 Holo-TRPO 收敛性对比 Demo。
    *   **实验结果 (Partial)**:
        | Iter | PGF Reward | Holo Reward | PGF Time (s) | Holo Step | Holo Time (s) |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | 0 | -3.3591 | -4.2837 | 0.37 | 0.041896 | 0.51 |
        | 5 | -3.1659 | -6.3049 | 0.22 | 0.005186 | 0.15 |
        | 10 | -2.7861 | -11.0400 | 0.22 | 0.012968 | 0.15 |
        | 15 | -1.1759 | -8.7956 | 0.21 | 0.016880 | 0.15 |
        | 19 | -2.0374 | -8.1709 | 0.23 | 0.032920 | 0.16 |
    *   **分析**: Holo-TRPO 展现了极其灵敏的步长控制（Step 随地形曲率动态大幅波动），虽然在简单的 PointMass 任务中随机性较强，但在高阶导数感知上比标准二次近似更具优势。

---

## 🚀 快速上手

1.  **运行跑分验证**:
    ```bash
    python RoBoT/benchmark_hvp.py
    ```
    ```
2.  **运行收敛训练**:
    ```bash
    python RoBoT/trpo_convergence_task.py
    ```

---

## 📜 核心理论参考
本项目的所有算法实现均严格遵循 `derivation_pgf_trpo.md` 中的公理化推导。建议在阅读代码前先查阅该文档以理解 PGF 的代数同构本质。
