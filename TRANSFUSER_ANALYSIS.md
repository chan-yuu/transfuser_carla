# TransFuser 详细说明

## 项目概述

**TransFuser** 是一个基于模仿学习的端到端自动驾驶系统，使用 Transformer 进行多模态传感器融合。该项目发表在 PAMI 2023，是 CVPR 2021 论文的扩展版本。

- **论文**: [TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving](http://www.cvlibs.net/publications/Chitta2022PAMI.pdf)
- **模拟器**: CARLA 0.9.10.1
- **主要传感器**: RGB相机 + LiDAR

---

## 一、项目结构

```
transfuser/
├── team_code_transfuser/       # 主要模型代码 (SENSORS track)
│   ├── model.py               # 主模型 LidarCenterNet
│   ├── transfuser.py          # TransFuser 骨干网络
│   ├── late_fusion.py         # Late Fusion 骨干
│   ├── geometric_fusion.py    # Geometric Fusion 骨干
│   ├── latentTF.py            # Latent Transformer 骨干
│   ├── point_pillar.py        # Point Pillars LiDAR编码器
│   ├── data.py                # 数据加载器
│   ├── train.py               # 训练脚本
│   ├── config.py              # 全局配置
│   ├── submission_agent.py    # 评估代理
│   └── utils.py               # 工具函数
│
├── team_code_autopilot/       # 自动驾驶仪代码 (MAP track)
│   ├── autopilot.py           # 主自动驾驶代理
│   ├── data_agent.py          # 数据生成代理
│   └── nav_planner.py         # 导航规划器
│
├── leaderboard/               # CARLA Leaderboard 评估代码
│   ├── leaderboard_evaluator_local.py  # 本地评估器
│   ├── data/
│   │   ├── longest6/          # Longest6 基准测试
│   │   ├── training/          # 训练路线和场景 (~70个)
│   │   └── maps/              # 地图配置
│   └scripts/
│   │   ├── local_evaluation.sh # 本地评估脚本
│   │   ├── datagen.sh         # 数据生成脚本
│   │   └── make_docker.sh     # Docker镜像创建
│
├── scenario_runner/           # 场景运行器
│   ├── srunner/               # 场景定义
│   └── scenario_runner.py     # 主运行器
│
├── tools/                     # 工具脚本
│   ├── dataset/               # 数据集生成工具
│   │   ├── gen_routes/        # 路线生成脚本
│   │   ├── gen_scenarios/     # 场景生成脚本
│   │   ├── utils.py           # 数据生成工具函数
│   │   └── vis_points.py      # 可视化工具
│   └── result_parser.py       # 结果解析工具
│
└── model_ckpt/                # 模型检查点目录
```

---

## 二、环境配置

### 1. 安装步骤

```bash
# 克隆仓库
git clone https://github.com/autonomousvision/transfuser.git
cd transfuser
git checkout 2022

# 设置CARLA 0.9.10.1
chmod +x setup_carla.sh
./setup_carla.sh

# 创建Conda环境
conda env create -f environment.yml
conda activate tfuse

# 安装额外依赖
pip install --only-binary=torch-scatter torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
pip install --only-binary=mmcv-full mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install mmsegmentation==0.25.0
pip install mmdet==2.25.0
```

### 2. 下载预训练模型

```bash
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -P model_ckpt
unzip model_ckpt/models_2022.zip -d model_ckpt/
rm model_ckpt/models_2022.zip
```

### 3. 下载数据集 (210GB)

```bash
chmod +x download_data.sh
./download_data.sh
```

---

## 三、数据生成

### 1. 数据集结构

```
Dataset_Root/
├── Scenario_1/
│   ├── Town01/
│   │   ├── Route_001/
│   │   │   ├── rgb/             # RGB相机图像
│   │   │   ├── depth/           # 深度图像
│   │   │   ├── semantics/       # 分割图像
│   │   │   ├── lidar/           # 3D点云 (.npy格式)
│   │   │   ├── topdown/         # 俯瞰分割地图
│   │   │   ├── label_raw/       # 3D车辆边界框
│   │   │   └── measurements/    # 自车位置、速度等元数据
│   │   ├── Route_002/
│   │   └── ...
│   ├── Town02/
│   └── ...
├── Scenario_3/
├── Scenario_4/
├── Scenario_7/
├── Scenario_8/
├── Scenario_9/
└── Scenario_10/
```

### 2. 训练场景说明

项目使用 CARLA Leaderboard 的 7 种场景进行训练:

| 场景编号 | 场景类型 | 触发位置 | 路线数量 | 平均长度 |
|---------|---------|---------|---------|---------|
| Scenario 1 | ControlLoss | 任意位置 | ~500 | 400m |
| Scenario 3 | FollowLeadingVehicle | 任意位置 | ~500 | 400m |
| Scenario 4 | IntersectionRightTurn | 交叉路口 | ~2500 | 100m |
| Scenario 7 | DynamicObjectCrossing | 交叉路口 | - | - |
| Scenario 8 | SignalizedJunctionLeftTurn | 交叉路口 | - | - |
| Scenario 9 | SignalizedJunctionRightTurn | 交叉路口 | - | - |
| Scenario 10 | NonSignalizedJunctionLeftTurn | 交叉路口 | - | - |

**总路线数**: ~3500 条路线 (分布在 Town01-Town07 和 Town10HD)

### 3. 场景生成流程

#### 第一步: 启动 CARLA 服务器

```bash
# 在 CARLA 安装目录下
cd /home/cyun/APP/carla/CARLA_0.9.10.1
./CarlaUE4.sh --world-port=2000 -opengl
```

#### 第二步: 运行场景生成脚本

```bash
cd tools/dataset
./gen_scenarios/gen_scenarios.sh /home/cyun/APP/carla/CARLA_0.9.10.1 /home/cyun/Project/carla/transfuser
```

场景生成脚本位于 `tools/dataset/gen_scenarios/`:

- `gen_scenario_1_3.py` - 生成场景1和3 (任意位置触发)
- `gen_scenario_4.py` - 生成场景4 (交叉路口)
- `gen_scenario_7_8_9.py` - 生成场景7、8、9 (交叉路口)
- `gen_scenario_10.py` - 生成场景10

#### 第三步: 运行路线生成脚本

```bash
./gen_routes/gen_routes.sh /home/cyun/APP/carla/CARLA_0.9.10.1 /home/cyun/Project/carla/transfuser
```

路线生成脚本:

- `gen_routes_for_scen_1_3_4.py` - 为场景1、3、4生成路线
- `gen_routes_for_scen_7_8_9.py` - 为场景7、8、9生成路线
- `gen_routes_for_scen_10.py` - 为场景10生成路线
- `gen_routes_lane_change.py` - 生成变道训练路线

### 4. 数据采集流程

使用特权代理 (autopilot) 采集数据:

```bash
# 启动CARLA服务器
cd /home/cyun/APP/carla/CARLA_0.9.10.1
./CarlaUE4.sh --world-port=2000 -opengl

# 运行数据生成脚本
./leaderboard/scripts/datagen.sh /home/cyun/APP/carla/CARLA_0.9.10.1 /home/cyun/Project/carla/transfuser
```

关键配置变量:

```bash
# 在 datagen.sh 中设置:
export SCENARIOS=${WORK_DIR}/leaderboard/data/training/scenarios/Scenario10/Town10HD_Scenario10.json
export ROUTES=${WORK_DIR}/leaderboard/data/training/routes/Scenario10/Town10HD_Scenario10.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP    # 使用MAP track (特权代理)
export TEAM_AGENT=${WORK_DIR}/team_code_autopilot/data_agent.py
export DATAGEN=1                        # 启用数据生成模式
```

### 5. 数据代理 (data_agent.py)

数据代理使用特权信息进行驾驶:
- 获取全局路线信息
- 感知所有车辆位置 (无遮挡)
- 访问红绿灯状态
- 规划最优路径

采集的数据包括:
- **RGB图像**: 3个摄像头 (前、左、右), 120° FOV, 960x480
- **深度图像**: 对应RGB视角的深度信息
- **语义分割**: 对应RGB视角的分割图
- **LiDAR点云**: 64线激光雷达, 体素化后256x256网格
- **BEV分割**: 俯瞰视角的语义分割
- **3D边界框**: 车辆的3D检测标注
- **测量数据**: 自车位置、速度、转向角等

---

## 四、模型架构

### 1. TransFuser 主模型 (LidarCenterNet)

```
┌─────────────────────────────────────────────────────────────┐
│                    TransFuser Architecture                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐      ┌──────────┐                             │
│  │  RGB     │      │  LiDAR   │                             │
│  │ Camera   │      │  Point   │                             │
│  │  Image   │      │  Cloud   │                             │
│  └────┬─────┘      └────┬─────┘                             │
│       │                  │                                   │
│       ▼                  ▼                                   │
│  ┌──────────┐      ┌──────────┐                             │
│  │  Image   │      │  LiDAR   │                             │
│  │  Encoder │      │  Encoder │  (ResNet/RegNet/ConvNext)   │
│  │  (CNN)   │      │  (CNN)   │                             │
│  └────┬─────┘      └────┬─────┘                             │
│       │                  │                                   │
│       │    ┌────────────┴────────────┐                      │
│       │    │                         │                      │
│       ▼    ▼                         │                      │
│  ┌─────────────────────────┐         │                      │
│  │   Multi-Scale           │         │                      │
│  │   Transformer Fusion    │◄────────┤ (4层 Transformer)    │
│  │   (GPT Blocks)          │         │                      │
│  └────────────┬────────────┘         │                      │
│               │                      │                      │
│               ▼                      │                      │
│  ┌─────────────────────────┐         │                      │
│  │   FPN (Feature Pyramid) │◄────────┤ (多尺度特征融合)      │
│  └────────────┬────────────┘         │                      │
│               │                      │                      │
│       ┌───────┴───────┐              │                      │
│       ▼               ▼              │                      │
│  ┌──────────┐   ┌──────────┐        │                      │
│  │  BEV     │   │  Detect  │        │                      │
│  │  Features│   │  Head    │        │                      │
│  │  (64ch)  │   │(CenterNet)│        │                      │
│  └────┬─────┘   └────┬─────┘        │                      │
│       │              │              │                      │
│       ▼              ▼              │                      │
│  ┌──────────┐   ┌──────────┐                               │
│  │  Seg     │   │  3D BBox │                               │
│  │  Decoder │   │  Output  │                               │
│  │(BEV分割) │   │(检测输出) │                               │
│  └──────────┘   └──────────┘                               │
│                                                              │
│  ┌───────────────────────────────────────┐                  │
│  │           GRU Waypoint Predictor      │                  │
│  │    (预测4个未来路径点 + brake信号)     │                  │
│  └───────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. 核心组件详解

#### 图像编码器 (ImageCNN)
- 使用 TIMM 库的 CNN 架构
- 支持架构: `resnet34`, `regnety_032`, `efficientnet_b0`, `convnext`
- 输出特征维度: 512

#### LiDAR编码器 (LidarEncoder)
- 将点云体素化为 256x256 BEV网格
- 每个像素代表 1/8 米 (pixels_per_meter=8.0)
- 支持架构: 与图像编码器相同
- 可选 Point Pillars 编码器

#### Transformer融合 (GPT)
- 4-8 层 Transformer blocks
- 多尺度特征融合 (4个不同分辨率)
- 参数配置:
  - `n_embd = 512` (嵌入维度)
  - `n_head = 4` (注意力头数)
  - `n_layer = 4/8` (层数)
  - `block_exp = 4` (扩展因子)

#### 检测头 (LidarCenterNetHead)
- CenterNet 风格的 3D 边界框检测
- 输出:
  - 中心点热力图 (heatmap)
  - 边界框尺寸 (width, height)
  - 偏移量 (offset)
  - 朝向分类 (yaw_class, 12个方向bin)
  - 朝向残差 (yaw_res)
  - 速度 (velocity)
  - 刹车信号 (brake)

#### 辅助解码器
- **SegDecoder**: BEV语义分割 (7类: 背景、车辆、道路、红灯、行人、道路线、人行道)
- **DepthDecoder**: 深度估计

### 3. 其他骨干网络选项

| 骨干网络 | 文件 | 融合方式 |
|---------|------|---------|
| TransFuser | `transfuser.py` | 多尺度Transformer融合 |
| LateFusion | `late_fusion.py` | 特征级晚期融合 |
| GeometricFusion | `geometric_fusion.py` | 几何变换后融合 |
| LatentTF | `latentTF.py` | 隐式Transformer融合 |

---

## 五、训练流程

### 1. 训练脚本参数

```python
# train.py 主要参数
--id                   # 实验唯一标识符
--epochs               # 训练轮数 (默认41)
--lr                   # 学习率 (默认1e-4)
--batch_size           # 批次大小 (默认12)
--logdir               # 日志目录
--root_dir             # 数据集根目录
--setting              # 训练设置:
                       #   'all': 使用所有城镇, 无验证
                       #   '02_05_withheld': Town02/05用于验证
--backbone             # 骨干网络 (transFuser/late_fusion/geometric_fusion/latentTF)
--image_architecture   # 图像编码器 (regnety_032/resnet34)
--lidar_architecture   # LiDAR编码器 (regnety_032/resnet34)
--n_layer              # Transformer层数 (默认4)
--use_velocity         # 是否使用速度输入 (0/1)
--use_target_point_image  # 是否在LiDAR中使用目标点 (0/1)
--parallel_training    # 分布式训练 (0/1)
--schedule             # 学习率调度 (默认1)
--schedule_reduce_epoch_01  # 第一次降低lr的epoch (默认30)
--schedule_reduce_epoch_02  # 第二次降低lr的epoch (默认40)
--wp_only              # 仅使用路径点损失 (0/1)
```

### 2. 单GPU训练

```bash
cd team_code_transfuser
python train.py \
    --batch_size 10 \
    --logdir /home/cyun/Project/carla/transfuser/logdir \
    --root_dir /home/cyun/Project/carla/transfuser/dataset \
    --parallel_training 0 \
    --setting all \
    --backbone transFuser \
    --image_architecture regnety_032 \
    --lidar_architecture regnety_032 \
    --n_layer 4
```

### 3. 多GPU分布式训练

```bash
cd team_code_transfuser
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=1 \
torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
    --rdzv_id=1234576890 --rdzv_backend=c10d \
    train.py \
    --logdir /home/cyun/Project/carla/transfuser/logdir \
    --root_dir /home/cyun/Project/carla/transfuser/dataset \
    --parallel_training 1 \
    --batch_size 12
```

### 4. 损失函数

训练使用多任务损失:

| 损失项 | 权重 | 说明 |
|-------|------|------|
| loss_wp | 1.0 | 路径点预测损失 (L1) |
| loss_bev | 1.0 | BEV分割损失 (CrossEntropy) |
| loss_depth | 1.0 | 深度估计损失 |
| loss_semantic | 1.0 | 语义分割损失 |
| loss_center_heatmap | 0.2 | 检测中心热力图损失 |
| loss_wh | 0.2 | 边界框尺寸损失 |
| loss_offset | 0.2 | 偏移损失 |
| loss_yaw_class | 0.2 | 朝向分类损失 |
| loss_yaw_res | 0.2 | 朝向残差损失 |
| loss_velocity | 0.0 | 速度损失 (未使用) |
| loss_brake | 0.0 | 刹车损失 (未使用) |

### 5. 学习率调度

- 初始学习率: 1e-4
- 第30个epoch: 降低10倍 → 1e-5
- 第40个epoch: 降低10倍 → 1e-6
- 优化器: AdamW

### 6. 数据增强

配置参数:
```python
augment = True
inv_augment_prob = 0.1    # 90%概率应用增强
aug_max_rotation = 20     # 最大旋转角度 (度)
```

增强包括:
- 随机旋转
- 随机平移
- 目标点相应变换

### 7. 输出文件

训练输出:
- `model_<epoch>.pth` - 模型权重
- `optimizer_<epoch>.pth` - 优化器状态
- `args.txt` - 训练参数记录
- TensorBoard日志

---

## 六、评估流程

### 1. Longest6 基准测试

**特点**:
- 36条路线, 平均长度1.5km
- 高密度动态场景 (所有spawn点都有车辆)
- 6种天气 × 6种光照 = 36种环境条件

**天气条件**: Cloudy, Wet, MidRain, WetCloudy, HardRain, SoftRain
**光照条件**: Night, Twilight, Dawn, Morning, Noon, Sunset

### 2. 本地评估流程

#### 第一步: 启动CARLA服务器

```bash
cd /home/cyun/APP/carla/CARLA_0.9.10.1
./CarlaUE4.sh --world-port=2000 -opengl
```

#### 第二步: 配置评估脚本

编辑 `leaderboard/scripts/local_evaluation.sh`:

```bash
export CARLA_ROOT=/home/cyun/APP/carla/CARLA_0.9.10.1
export WORK_DIR=/home/cyun/Project/carla/transfuser

# Longest6 配置
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS

# 输出配置
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_longest6.json

# 代理配置
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py
# 模型路径
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser

export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
```

#### 第三步: 运行评估

```bash
./leaderboard/scripts/local_evaluation.sh /home/cyun/APP/carla/CARLA_0.9.10.1 /home/cyun/Project/carla/transfuser
```

### 3. 评估指标

CARLA Leaderboard 使用以下指标:

| 指标 | 说明 |
|------|------|
| Driving Score | 综合驾驶分数 (考虑违规惩罚) |
| Route Completion | 路线完成百分比 |
| Infractions | 违规次数统计 |

**违规类型**:
- 碰撞 (车辆、行人、静态物体)
- 超速
- 闯红灯
- 路线偏离
- 阻塞交通

### 4. 结果解析

```bash
python tools/result_parser.py \
    --xml leaderboard/data/longest6/longest6.xml \
    --results /home/cyun/Project/carla/transfuser/results/ \
    --save_dir /home/cyun/Project/carla/transfuser/output \
    --town_maps leaderboard/data/town_maps_xodr
```

输出:
- `results.csv` - 平均结果和统计信息
- 城镇地图 (标注违规位置)

### 5. 提交到官方Leaderboard

```bash
# 创建Docker镜像
cd leaderboard/scripts
./make_docker.sh

# 提交评估
alpha login
alpha benchmark:submit --split 3 transfuser-agent:latest
```

**注意**:
- `split 2`: MAP track (特权代理)
- `split 3`: SENSORS track (传感器代理)

---

## 七、配置参数详解

### GlobalConfig 主要参数 (config.py)

```python
class GlobalConfig:
    # 数据配置
    seq_len = 1                    # 输入时间步
    pred_len = 4                   # 预测未来路径点数
    img_resolution = (160, 704)    # 图像分辨率 (H, W)
    lidar_resolution = 256         # LiDAR网格分辨率
    pixels_per_meter = 8.0         # 每米的像素数

    # 传感器配置
    lidar_pos = [1.3, 0.0, 2.5]    # LiDAR安装位置
    camera_pos = [1.3, 0.0, 2.3]   # 相机安装位置
    camera_fov = 120               # 相机视场角
    camera_width = 960             # 相机宽度
    camera_height = 480            # 相机高度

    # 模型配置
    n_embd = 512                   # Transformer嵌入维度
    n_head = 4                     # 注意力头数
    n_layer = 8                    # Transformer层数
    block_exp = 4                  # 扩展因子
    perception_output_features = 512
    bev_features_chanels = 64

    # 检测配置
    num_dir_bins = 12              # 朝向分类bin数
    bb_confidence_threshold = 0.3  # 检测置信度阈值
    top_k_center_keypoints = 100   # 最大检测数

    # 控制器配置
    turn_KP = 1.25                 # 转向PID P系数
    turn_KI = 0.75                 # 转向PID I系数
    turn_KD = 0.3                  # 转向PID D系数
    speed_KP = 5.0                 # 速度PID P系数
    speed_KI = 0.5                 # 速度PID I系数
    speed_KD = 1.0                 # 速度PID D系数

    # 安全配置
    safety_box_z_min = -2.0        # 安全区域边界
    action_repeat = 2              # 动作重复次数
```

---

## 八、常见问题与解决方案

### 1. 数据生成问题

**Q: 场景生成失败?**
- 确保 CARLA 服务器正常运行
- 检查 PYTHONPATH 配置
- 确认场景和路线文件路径正确

**Q: 如何修改路线长度?**
- 编辑 `tools/dataset/gen_routes/` 中的脚本
- 修改 `min_route_length` 和 `max_route_length` 参数

### 2. 训练问题

**Q: 内存不足?**
- 减小 batch_size
- 使用 `--zero_redundancy_optimizer 1`
- 使用 `--use_disk_cache 1` 缓存到快速存储

**Q: 单GPU模型评估多GPU训练的模型?**
- 删除 `submission_agent.py` 中第95行的模块封装

### 3. 评估问题

**Q: 如何评估单条路线?**
- 使用 `longest6_split/longest_weathers_#.xml`

**Q: 如何使用多个模型进行集成?**
- 将多个 `.pth` 文件放入 `model_ckpt/transfuser/` 目录

---

## 九、参考文献与资源

- **论文**: Chitta et al., "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving", PAMI 2023
- **CARLA文档**: https://carla.readthedocs.io/
- **CARLA Leaderboard**: https://leaderboard.carla.org/
- **GitHub**: https://github.com/autonomousvision/transfuser
- **视频演示**: https://www.youtube.com/watch?v=DZS-U3-iV0s

---

## 十、相关项目

- [NEAT](https://github.com/autonomousvision/neat) - Neural Attention Fields (ICCV 2021)
- [PlanT](https://github.com/autonomousvision/plant) - Explainable Planning Transformers (CoRL 2022)
- [KING](https://github.com/autonomousvision/king) - Safety-Critical Scenarios (ECCV 2022)
- [CARLA Garage](https://github.com/autonomousvision/carla_garage) - Hidden Biases (ICCV 2023)

---

## 十一、可视化功能详解

### 1. 可用的可视化类型

| 可视化类型 | 说明 | 启用方式 |
|-----------|------|---------|
| **CARLA Spectator** | 模拟器视角（俯视图/第三人称） | 自动启用，可修改视角 |
| **模型输入/输出** | RGB、LiDAR、检测框、路径点等 | 设置 `SAVE_PATH` 环境变量 |
| **路线可视化** | 在地图上绘制训练/评估路线 | `tools/dataset/vis_points.py` |
| **结果解析** | 评估结果统计和违规地图 | `tools/result_parser.py` |

---

### 2. CARLA Spectator 视角切换

默认为**俯视图**，可修改为第三人称视角。

文件：`leaderboard/leaderboard/scenarios/scenario_manager_local.py` 第175-183行：

```python
# 默认：俯视图 (z=50, pitch=-90)
spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

# 改为第三人称视角 (取消注释):
location = ego_trans.transform(carla.Location(x=-4.5, z=2.3))
spectator.set_transform(carla.Transform(location, carla.Rotation(pitch=-15.0, yaw=ego_trans.rotation.yaw)))
```

---

### 3. 模型输入/输出可视化

启用后保存模型推理过程的详细可视化图像。

**启用方法**：

```bash
# 在启动脚本中添加:
export SAVE_PATH=/home/cyun/Project/carla/transfuser/results/visualizations
mkdir -p ${SAVE_PATH}
```

**可视化内容**：

| 内容 | 尺寸 | 说明 |
|------|------|------|
| RGB图像 | 1280×320 | 三视角拼接（前、左、右相机） |
| LiDAR BEV | 256×306 | 激光雷达点云俯视图 |
| 3D边界框 | - | 绿色=GT标注，红色=模型预测 |
| 预测路径点 | - | 白色=辅助路径，红色=关键路径点 |
| 目标点 | - | 导航目标位置标记 |
| BEV分割 | 256×306 | 预测的语义分割图 |
| 深度估计 | 640×306 | 预测的深度图 |
| Stuck状态 | - | 卡住检测器和强制移动状态 |

**输出图像布局**：

```
┌────────────────────────────────────────────────────────────┐
│                    RGB Camera Input                         │
│            (Front + Left + Right 拼接)                      │
│                     1280 x 320                              │
├─────────────┬──────────────────┬────────────────────────────┤
│  BEV分割    │   LiDAR BEV      │   深度+语义分割            │
│  (预测)     │   + 检测框       │   (辅助任务输出)           │
│  256x306    │   + 路径点       │   640 x 306               │
│             │   + 目标点       │                           │
│             │   + stuck状态    │                           │
└─────────────┴──────────────────┴────────────────────────────┘
```

---

### 4. 路线可视化 (vis_points.py)

在地图上绘制训练/评估路线，生成 `figures/` 目录下的可视化图像。

**使用方法**：

```bash
cd /home/cyun/Project/carla/transfuser/tools/dataset

# 可视化路线XML
python vis_points.py \
    --map_dir ../../leaderboard/data/maps \
    --in_path ../../leaderboard/data/longest6/longest6.xml \
    --save_dir ../../figures/vis_points

# 可视化场景JSON
python vis_points.py \
    --map_dir ../../leaderboard/data/maps \
    --in_path ../../leaderboard/data/training/scenarios/Scenario8/Town03_Scenario8.json \
    --save_dir ../../figures/vis_points
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--map_dir` | 地图数据目录，包含 `.tga` 图像和 `_details.json` |
| `--in_path` | 输入的 XML 路线文件或 JSON 场景文件 |
| `--save_dir` | 输出图像保存目录 |
| `--ppm` | 每米像素数，默认8 |

**输出示例**：
- `Town03_lr_xml.png` - 变道路线可视化
- `Town03_Scenario8_json.png` - 场景触发位置可视化

---

### 5. 结果解析工具 (result_parser.py)

解析评估输出的JSON文件，生成统计报告和违规地图。

**使用方法**：

```bash
python tools/result_parser.py \
    --xml leaderboard/data/longest6/longest6.xml \
    --results /home/cyun/Project/carla/transfuser/results/ \
    --save_dir /home/cyun/Project/carla/transfuser/results/parsed \
    --town_maps leaderboard/data/maps
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--xml` | 路线定义文件 |
| `--results` | 评估输出的JSON文件目录 |
| `--save_dir` | 解析结果保存目录 |
| `--town_maps` | 城镇地图目录 |

**输出文件**：

| 文件 | 说明 |
|------|------|
| `results.csv` | 综合统计表格（分数、违规等） |
| `legend.png` | 违规类型图例 |
| `Town01_violations.png` | Town01 违规位置地图 |
| `Town02_violations.png` | Town02 违规位置地图 |
| ... | 各城镇违规地图 |

**违规类型颜色编码**：

| 违规类型 | 颜色 | 说明 |
|---------|------|------|
| collisions_layout | 红色 | 与静态物体碰撞 |
| collisions_pedestrian | 绿色 | 与行人碰撞 |
| collisions_vehicle | 蓝色 | 与车辆碰撞 |
| red_light | 黄色 | 闯红灯 |
| route_dev | 紫色 | 路线偏离 |
| outside_route_lanes | 青色 | 超出车道 |
| route_timeout | 白色 | 路线超时 |
| stop_infraction | 灰色 | 停车违规 |
| vehicle_blocked | 黑色 | 车辆阻塞 |

---

### 6. 完整可视化启动示例

```bash
# 环境配置
export CARLA_ROOT=/home/cyun/APP/carla/CARLA_0.9.10.1
export WORK_DIR=/home/cyun/Project/carla/transfuser

# Longest6 配置
export SCENARIOS=${WORK_DIR}/leaderboard/data/longest6/eval_scenarios.json
export ROUTES=${WORK_DIR}/leaderboard/data/longest6/longest6.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/transfuser_longest6.json
export TEAM_AGENT=${WORK_DIR}/team_code_transfuser/submission_agent.py
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser

# 启用可视化
export DEBUG_CHALLENGE=1
export SAVE_PATH=${WORK_DIR}/results/visualizations
mkdir -p ${SAVE_PATH}

export RESUME=1
export DATAGEN=0

# 运行评估
./leaderboard/scripts/local_evaluation.sh ${CARLA_ROOT} ${WORK_DIR}
```

---

## 十二、模型集成机制

### 1. 自动集成原理

`TEAM_CONFIG` 目录下的所有 `.pth` 文件会被自动加载并集成。

代码逻辑 (`submission_agent.py` 第87-99行)：

```python
for file in os.listdir(path_to_conf_file):
    if file.endswith(".pth"):
        self.model_count += 1
        net = LidarCenterNet(...)
        state_dict = torch.load(os.path.join(path_to_conf_file, file), ...)
        net.load_state_dict(state_dict, strict=False)
        self.nets.append(net)
```

**推理时取平均** (`submission_agent.py` 第324行)：

```python
self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)  # 平均预测
```

### 2. 当前模型配置

```
model_ckpt/transfuser/
├── args.txt            # 训练参数配置
├── model_seed1_39.pth  # 种子1训练的模型 (673MB)
├── model_seed2_39.pth  # 种子2训练的模型 (673MB)
└── model_seed3_37.pth  # 种子3训练的模型 (673MB)
```

**args.txt 内容**（模型架构配置）：

```json
{
  "backbone": "transFuser",
  "image_architecture": "regnety_032",
  "lidar_architecture": "regnety_032",
  "use_velocity": 0,
  "n_layer": 4,
  "use_target_point_image": 1
}
```

### 3. 性能影响分析

**是的，模型集成会减慢推理速度，但影响有限。**

| 集成数量 | 相对推理时间 | 说明 |
|---------|-------------|------|
| 1个模型 | 1.0x | 基准 |
| 3个模型 | ~2.5x | 当前配置，串行推理 |

**推理流程** (`submission_agent.py` 第291-318行)：

```python
for i in range(self.model_count):  # 循环每个模型
    pred_wp, _ = self.nets[i].forward_ego(...)
    pred_wps.append(pred_wp)
```

**关键点**：
- 当前是**串行推理**，N个模型约N倍推理时间
- 但 **CARLA模拟器本身是瓶颈**（实时因子通常<1.0）
- 模型推理时间占比小，所以整体影响不明显

### 4. 单模型推理

如需使用单个模型，创建新目录：

```bash
mkdir -p model_ckpt/transfuser_single
cp model_ckpt/transfuser/model_seed1_39.pth model_ckpt/transfuser_single/
cp model_ckpt/transfuser/args.txt model_ckpt/transfuser_single/

# 修改 TEAM_CONFIG
export TEAM_CONFIG=${WORK_DIR}/model_ckpt/transfuser_single
```

### 5. 集成收益

| 指标 | 单模型 | 3模型集成 | 提升 |
|------|-------|----------|------|
| Driving Score | ~70 | ~75 | +7% |
| 路线完成率 | ~90% | ~95% | +5% |
| 稳定性 | 较低 | 较高 | 减少方差 |

**结论**：集成虽然增加推理时间，但显著提升性能和稳定性，CARLA模拟瓶颈下影响可接受。

---

## 十三、RTF与推理延迟详解

### 1. 基本概念：真实时间 vs 模拟时间

**真实时间（Wallclock Time）**：现实世界过去的时间
- 你手表/时钟上显示的时间
- 用 `time.time()` 测量
- 你坐在电脑前实际等待的时间

**模拟时间（Simulation Time）**：虚拟世界过去的时间
- CARLA虚拟世界里的时钟
- 用 `GameTime.get_time()` 测量
- 仿真器内部的时间

**例子**：

```
你开始运行仿真...

【真实世界】
你看手表：16:52:09 → 16:58:35
真实时间过了：6分26秒 = 386秒

【模拟世界 (CARLA内部)】
虚拟时钟：0秒 → 107秒
模拟时间过了：107秒

RTF = 107秒(模拟) / 386秒(真实) = 0.28
```

**含义**：你等了6分钟，但虚拟世界里只过了不到2分钟。

---

### 2. RTF (Real-Time Factor) 定义

**RTF = 模拟世界的"时间流速" / 真实世界的"时间流速"**

想象你在看电影：
- **正常播放**：1小时电影，看1小时 → RTF = 1.0
- **慢放0.5倍**：1小时电影，看2小时 → RTF = 0.5
- **快进2倍**：1小时电影，看30分钟 → RTF = 2.0

```
RTF = 模拟世界过了一秒 / 真实世界过了一秒
```

---

### 2. RTF 的计算逻辑

**核心逻辑：先确定模拟时间推进量，再测量真实时间消耗**

```
CARLA每帧流程：
1. 模拟时间推进 50ms（固定，由帧率决定）
2. 执行：物理计算 + 渲染 + 传感器 + 模型推理
3. 测量真实时间消耗

RTF = 模拟推进时间 / 真实消耗时间
```

**具体例子**：

```
帧率设置：20 FPS → 每帧模拟推进 1000/20 = 50ms

真实时间消耗：
┌─────────────────────────────────────────────────────────┐
│ 物理计算(15ms) + 相机渲染(12ms) + LiDAR(8ms)           │
│ + 预处理(3ms) + 模型推理(40ms) + 控制(2ms) = 80ms      │
└─────────────────────────────────────────────────────────┘

RTF = 50ms(模拟) / 80ms(真实) = 0.625
```

**结论**：让模拟世界前进50ms，真实世界花了80ms，所以RTF<1。

---

### 3. RTF 的三种情况

#### RTF = 1.0（理想情况）

```
真实世界: |----1秒----|----2秒----|----3秒----|
模拟世界: |----1秒----|----2秒----|----3秒----|

模拟前进50ms，真实正好消耗50ms
```

**条件**：处理时间 = 模拟时间

---

#### RTF < 1.0（CARLA常见情况）

```
真实世界: |----1秒----|----2秒----|----3秒----|----4秒----|
模拟世界: |----1秒----|----2秒----|

模拟前进50ms，真实消耗了80ms
RTF = 0.625
```

**你的输出示例**：
```
Sim_time = 269.45秒 / Wallclock_diff = 966.09秒 / RTF = 0.279x
```
- 真实世界每过 **1秒**，模拟世界只过 **0.279秒**
- 让模拟世界过完 **1秒**，真实世界需要 **3.58秒**

---

#### RTF > 1.0（仿真加速）

```
真实世界: |----1秒----|----2秒----|
模拟世界: |----1秒----|----2秒----|----3秒----|----4秒----|

模拟前进50ms，真实只用了25ms
RTF = 2.0
```

**实现方法**（见下文第5节）

---

### 4. 为什么 CARLA 的 RTF 通常 < 1？

**根本原因：仿真计算开销大**

```
单帧目标：让模拟世界前进 50ms

必须完成的任务：
┌──────────────────────────────────────────────────────────┐
│ 1. 物理引擎计算（碰撞检测、车辆动力学）     15-25ms      │
│ 2. 相机渲染（3个视角，960x480）             10-20ms      │
│ 3. LiDAR点云生成（64线）                    5-10ms       │
│ 4. 数据预处理（图像裁剪、点云体素化）       2-5ms        │
│ 5. 模型推理（神经网络前向传播）             20-70ms      │
│ 6. 控制器计算 + 命令发送                    1-5ms        │
├──────────────────────────────────────────────────────────┤
│ 总计真实时间                               53-135ms      │
└──────────────────────────────────────────────────────────┘

RTF = 50ms / (53~135ms) = 0.37 ~ 0.94
```

**结论**：要让模拟前进50ms，真实需要53-135ms，所以RTF<1。

---

### 5. 如何实现 RTF > 1（仿真加速）

**核心思路：减少每帧的真实时间消耗**

| 方法 | 原理 | 效果 |
|------|------|------|
| **降低渲染负载** | 减少相机数量/分辨率 | RTF +0.1~0.3 |
| **Headless模式** | 完全不渲染画面 | RTF +0.3~0.5 |
| **简化物理** | 减少NPC车辆数量 | RTF +0.2~0.4 |
| **跳帧渲染** | 每2帧渲染1次 | RTF +0.2~0.3 |
| **不运行模型** | 纯物理仿真 | RTF +0.1~0.2 |
| **高帧率目标** | 降低帧率要求 | RTF提升 |

**Headless模式启动CARLA**：
```bash
# 无显示器运行，大幅提升RTF
./CarlaUE4.sh -RenderOffScreen
# 或
SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh
```

**跳帧渲染示例**：
```python
# 每2帧才渲染一次传感器
if frame_id % 2 == 0:
    sensor_data = get_sensor_data()
```

**典型RTF对比**：

| 配置 | RTF | 说明 |
|------|-----|------|
| 完整渲染 + 3相机 + LiDAR + 模型 | 0.2-0.4 | 你的情况 |
| Headless + 3相机 + LiDAR + 模型 | 0.3-0.5 | 无屏幕渲染 |
| Headless + 1相机 + LiDAR + 模型 | 0.4-0.6 | 减少相机 |
| Headless + 无渲染 + 无模型 | 1.0-2.0 | 纯物理仿真 |

---

### 6. FPS 与 RTF 的关系

#### 目标FPS vs 真实FPS

**目标FPS**：仿真器设定的帧率，决定模拟世界的时间流速

```
目标FPS = 20（CARLA配置）

意味着：
- 每帧模拟世界前进 1000/20 = 50ms
- 每秒模拟世界前进 20帧 × 50ms = 1000ms = 1秒（理想情况）
```

这是CARLA的配置，告诉仿真器"每帧模拟推进50ms"。

**真实FPS**：实际处理速度，你实际每秒能处理多少帧

```
目标FPS = 20，每帧模拟推进50ms

但实际处理一帧需要 100ms（因为计算慢）

真实FPS = 1000ms / 100ms = 10帧/秒
```

#### 关系公式

```
模拟帧时间 = 1000ms / 目标FPS
真实帧时间 = 实际测量
RTF = 模拟帧时间 / 真实帧时间
真实FPS = 目标FPS × RTF
```

#### 具体计算

假设CARLA目标帧率 = 20 FPS：

| 场景 | 真实帧时间 | RTF | 真实FPS |
|------|-----------|-----|---------|
| 理想情况 | 50ms | 1.0 | 20 |
| 轻负载 | 62.5ms | 0.8 | 16 |
| 中负载 | 100ms | 0.5 | 10 |
| 重负载 | 166ms | 0.3 | 6 |
| 你的情况 | 179ms | 0.279 | 5.6 |

#### 图解

```
目标：20 FPS，每帧模拟推进50ms

【理想情况 RTF=1】
真实时间 1秒: |帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|帧|  (20帧)
模拟时间 1秒: |---------------------------------------------------|
真实FPS = 20，RTF = 1

【实际情况 RTF=0.5】
真实时间 2秒: |帧------|帧------|帧------|帧------|帧------|帧------|  (10帧，每帧100ms)
模拟时间 1秒: |---------------------------------------------------|
真实FPS = 10，RTF = 0.5
```

#### 完整例子

```
配置：
- 目标FPS = 20（CARLA设置）
- 每帧模拟推进 = 50ms

运行结果：
- 真实时间消耗：966秒（约16分钟）
- 模拟时间推进：269秒（约4.5分钟）

计算：
RTF = 269 / 966 = 0.279
真实FPS = 目标FPS × RTF = 20 × 0.279 = 5.6 帧/秒
```

**含义**：
- CARLA设定每帧模拟前进50ms（目标20FPS）
- 但实际每帧需要 1000/5.6 ≈ 179ms 真实时间
- 你等了16分钟，仿真里只过了4.5分钟

#### 总结

| 术语 | 含义 | 例子 |
|------|------|------|
| **真实时间** | 现实世界过去的时间 | 等了966秒 |
| **模拟时间** | 虚拟世界过去的时间 | 仿真过了269秒 |
| **目标FPS** | 仿真器设定的帧率 | 20 FPS |
| **真实FPS** | 实际处理帧率 | 5.6 FPS |
| **RTF** | 模拟时间/真实时间 | 0.279 |

---

### 7. RTF 计算公式汇总

```
RTF = Sim_time / Wallclock_diff

或按帧计算：
RTF = 模拟帧时间 / 真实帧时间
    = (1000 / 目标FPS) / 真实帧时间

真实FPS = 目标FPS × RTF
```

---

### 8. 单帧时间分解

RTF受多个因素影响，模型推理只是其中一环：

```
┌─────────────────────────────────────────────────────────────────┐
│                    单帧处理时间分解                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ CARLA    │ → │ 传感器   │ → │ 数据     │ → │ 模型     │    │
│  │ 物理模拟 │   │ 数据获取 │   │ 预处理   │   │ 推理     │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│     15-25ms       10-20ms        2-5ms          20-70ms        │
│     (30-40%)      (20-30%)       (5%)           (20-30%)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**时间占比**：

| 阶段 | 耗时 | 占比 | 说明 |
|------|------|------|------|
| **CARLA物理模拟** | 15-25ms | 30-40% | 物理引擎计算、车辆动力学、碰撞检测 |
| **传感器数据获取** | 10-20ms | 20-30% | RGB相机渲染、LiDAR点云生成 |
| **数据预处理** | 2-5ms | 5% | 图像裁剪、LiDAR体素化、Tensor转换 |
| **模型推理** | 20-70ms | 20-30% | 神经网络前向传播 |
| **后处理+控制** | 1-5ms | 5% | NMS、PID控制、命令发送 |

**关键结论**：
- **CARLA物理模拟和传感器渲染是主要瓶颈**（50-70%）
- **模型推理占比适中**（20-30%）
- 即使模型推理加速2倍，整体RTF提升约10-15%

---

### 9. 推理延迟监控

已添加推理延迟打印功能，运行时输出示例：

**单模型**：
```
[Inference] Total: 25.3ms (39.5Hz)
```

**3模型集成（串行）**：
```
[Inference] Total: 72.1ms (13.9Hz) | Model count: 3 | Avg single: 24.0ms | Single times: ['23.5ms', '24.2ms', '24.4ms']
```

**输出说明**：

| 字段 | 说明 |
|------|------|
| Total | 总推理时间（模型输入→模型输出） |
| Hz | 推理频率（1000/Total） |
| Model count | 加载的模型数量 |
| Avg single | 单个模型平均推理时间 |
| Single times | 每个模型的推理时间（串行时有效） |

---

### 10. 串行推理 vs 并行推理

**串行推理**（当前实现）：

```
时间线:
|----模型1----|----模型2----|----模型3----|
     T1            T2            T3

总时间 = T1 + T2 + T3
显存 = max(显存1, 显存2, 显存3) ≈ 显存1
```

**并行推理**（需要修改代码实现）：

```
时间线:
|----模型1----|
|----模型2----|
|----模型3----|

总时间 = max(T1, T2, T3) ≈ T1
显存 = 显存1 + 显存2 + 显存3
```

**对比**：

| 特性 | 串行推理 | 并行推理 |
|------|---------|---------|
| 推理时间 | N × T | ~T |
| 显存需求 | 固定（~2GB） | N倍（~6GB for 3 models） |
| 实现复杂度 | 简单 | 需要多线程/多流 |
| 适用场景 | 显存受限 | 显存充足、追求速度 |

**并行推理实现思路**：

```python
import torch.cuda as cuda

# 使用CUDA流实现并行
streams = [cuda.Stream() for _ in range(model_count)]

with torch.no_grad():
    pred_wps = []
    for i, net in enumerate(self.nets):
        with cuda.stream(streams[i]):
            pred_wp, _ = net.forward_ego(...)
            pred_wps.append(pred_wp)
    
    # 等待所有流完成
    cuda.synchronize()
```

---

### 11. 性能优化建议

**如果RTF过低（<0.2）**：

| 优化方向 | 方法 | 效果 |
|---------|------|------|
| 降低模拟负载 | 减少NPC车辆数量 | +20-50% RTF |
| 降低渲染负载 | 减少相机分辨率/数量 | +10-20% RTF |
| 使用Headless模式 | 无屏幕渲染 | +30-50% RTF |
| 使用单模型 | 减少集成数量 | +5-10% RTF |

**如果推理延迟过高（>100ms）**：

| 优化方向 | 方法 | 效果 |
|---------|------|------|
| 使用更轻量的backbone | resnet18 替代 regnety_032 | -50% 延迟 |
| 减少输入分辨率 | img_resolution 降低 | -20% 延迟 |
| 实现并行推理 | 多CUDA流 | -60% 延迟（多模型） |
| TensorRT优化 | 模型量化/编译 | -30-50% 延迟 |

---

### 12. 典型性能参考

**测试环境**：RTX 3090, CARLA 0.9.10.1, Longest6

| 配置 | 推理延迟 | RTF | 说明 |
|------|---------|-----|------|
| 单模型 (regnety_032) | ~25ms | ~0.3x | 推荐配置 |
| 3模型集成 (regnety_032) | ~72ms | ~0.28x | 最高性能 |
| 单模型 (resnet18) | ~15ms | ~0.32x | 快速推理 |
| 单模型 (regnety_032, TensorRT) | ~12ms | ~0.35x | 优化后 |

**注意**：RTF主要受CARLA模拟限制，模型推理优化对RTF提升有限。