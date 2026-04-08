# TransFuser: 基于 Transformer 的多模态融合自动驾驶系统

## 📖 项目概述

**TransFuser** 是一个基于模仿学习（Imitation Learning）的端到端自动驾驶系统，发表于 PAMI 2023（CVPR 2021 扩展版）。该系统核心创新在于使用 **Transformer** 机制实现 RGB 图像与 LiDAR 点云的高效多模态传感器融合。

本仓库集成了模型训练、数据采集、本地评估以及结果解析的完整工作流，适配 **CARLA 0.9.10.1** 仿真环境。

---

## 📂 项目结构

```text
├── team_code_transfuser/       # 核心模型代码 (SENSORS track)
│   ├── model.py               # 主模型 LidarCenterNet (感知+规划)
│   ├── transfuser.py          # 基于 Transformer 的特征融合骨干网
│   ├── submission_agent.py    # 评估代理 (处理传感器输入并输出控制)
│   └── config.py              # 全局参数配置 (传感器位置、PID系数等)
├── team_code_autopilot/       # 特权代理代码 (用于大规模数据采集)
├── leaderboard/               # CARLA Leaderboard 评估框架
│   ├── leaderboard_evaluator_local.py  # 本地评估器核心逻辑
│   └── scripts/               # 运行评估与数据生成的 shell 脚本
├── scenario_runner/           # 场景运行器 (定义交通状况与违规检测)
├── tools/                     # 辅助工具
│   ├── dataset/               # 路线/场景生成及可视化工具
│   └── result_parser.py       # 评估结果解析与违规地图生成脚本
└── model_ckpt/                # 预训练模型权重存放目录
```

---

## 🧠 核心技术架构

### 1. 多模态融合模型 (LidarCenterNet)
*   **特征提取**：采用 CNN（如 RegNet/ResNet）作为图像和 LiDAR BEV 的编码器。
*   **Transformer 融合**：在 4 个不同尺度上应用 Transformer Block，实现图像语义与空间点云信息的交叉关注（Cross-attention）。
*   **多任务输出**：
    *   **目标检测**：CenterNet 风格的 3D 边界框预测。
    *   **路径规划**：基于 GRU 的路径点（Waypoints）序列预测。
    *   **辅助解码**：BEV 语义分割与深度图预测。

### 2. 评估工作流
系统通过 `run_evaluation.sh` 启动，执行路径如下：
1.  **初始化**：`LeaderboardEvaluator` 连接 CARLA 服务器并设置同步模式（20Hz）。
2.  **场景加载**：`RouteIndexer` 解析 XML 路线，`RouteScenario` 配置城镇与天气。
3.  **仿真循环**：`ScenarioManager` 驱动每一帧：
    *   获取传感器数据 -> 调用 `HybridAgent.run_step()` -> PID 控制器计算 -> 应用 `VehicleControl`。
4.  **模型集成**：支持加载多个 `.pth` 模型，通过对预测路径点取平均来增强驾驶稳定性。

---

## 🚀 关键操作指南

### 1. 环境准备
```bash
# 创建并激活环境
conda env create -f environment.yml
conda activate tfuse
# 运行 CARLA 服务器 (建议使用 -opengl 模式)
./CarlaUE4.sh --world-port=2000 -opengl
```

### 2. 本地评估 (Longest6 基准)
Longest6 包含 36 条长路线，覆盖高密度交通和 36 种天气/光照组合。
```bash
# 修改脚本中的路径配置后运行
./leaderboard/scripts/local_evaluation.sh
```

### 3. 数据采集
使用特权代理采集大规模训练集：
```bash
# 设置 DATAGEN=1 并运行
./leaderboard/scripts/datagen.sh
```

---

## 📊 性能分析与可视化

### 推理延迟与 RTF
*   **RTF (Real-Time Factor)**：模拟时间与真实时间之比。包含 3 相机 + LiDAR + 3模型集成时，RTF 约为 **0.27x - 0.3x**。
*   **推理耗时**：单模型推理约 **25ms** (39Hz)，3 模型集成约 **72ms** (14Hz)。

### 结果解析工具
使用 `tools/result_parser.py` 可以将 `results.json` 转换为直观的报告：
*   **违规统计**：碰撞、闯红灯、路线偏离等详细频率表。
*   **违规地图**：在城镇地图上标注所有事故发生的精确坐标。

### 调试视图
启用 `SAVE_PATH` 环境变量后，Agent 会保存每帧的可视化图像，包含：
*   前/左/右三相机拼接画面。
*   LiDAR BEV 点云与预测/GT 边界框对比。
*   预测路径点、目标点及 BEV 分割结果。

---

## 🔗 参考文献
*   **Paper**: [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving (PAMI 2023)](http://www.cvlibs.net/publications/Chitta2022PAMI.pdf)
*   **Official Repo**: [autonomousvision/transfuser](https://github.com/autonomousvision/transfuser)
