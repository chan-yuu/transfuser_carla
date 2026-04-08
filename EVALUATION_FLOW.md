# TransFuser 评估流程详解

## 一、评估脚本调用链

```
run_evaluation.sh
    │
    ▼
leaderboard_evaluator_local.py  (主评估器)
    │
    ├── RouteIndexer           (路线索引器，解析routes和scenarios)
    │
    ├── RouteScenario          (路线场景，加载场景配置)
    │
    ├── ScenarioManager        (场景管理器，控制仿真循环)
    │
    └── AgentWrapper           (代理包装器)
            │
            ▼
        submission_agent.py    (HybridAgent)
            │
            ├── setup()        (初始化模型)
            │
            └── run_step()     (每帧推理)
                    │
                    └── model.forward_ego()  (模型前向传播)
```

---

## 二、leaderboard_evaluator_local.py 执行流程

### 第1步: 初始化 (main函数)

```python
def main():
    # 1. 解析命令行参数
    arguments = parser.parse_args()
    
    # 2. 创建统计管理器
    statistics_manager = StatisticsManager()
    
    # 3. 创建评估器并运行
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    leaderboard_evaluator.run(arguments)
```

### 第2步: LeaderboardEvaluator.__init__()

```python
def __init__(self, args, statistics_manager):
    # 1. 连接CARLA服务器
    self.client = carla.Client(args.host, int(args.port))  # 默认localhost:2000
    
    # 2. 加载初始世界
    self.world = self.client.load_world('Town01')
    
    # 3. 获取TrafficManager
    self.traffic_manager = self.client.get_trafficmanager(8000)
    
    # 4. 动态导入Agent模块
    module_name = os.path.basename(args.agent).split('.')[0]  # "submission_agent"
    self.module_agent = importlib.import_module(module_name)
    
    # 5. 创建ScenarioManager
    self.manager = ScenarioManager(args.timeout, args.debug > 1)
```

### 第3步: LeaderboardEvaluator.run()

```python
def run(self, args):
    # 1. 创建路线索引器 (解析routes和scenarios文件)
    route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)
    
    # 2. 如果resume，从checkpoint恢复
    if args.resume:
        route_indexer.resume(args.checkpoint)
        self.statistics_manager.resume(args.checkpoint)
    
    # 3. 循环处理每条路线
    while route_indexer.peek():
        config = route_indexer.next()           # 获取下一条路线配置
        self._load_and_run_scenario(args, config)  # 运行该路线
        route_indexer.save_state(args.checkpoint)   # 保存进度
    
    # 4. 计算并保存全局统计
    global_stats_record = self.statistics_manager.compute_global_statistics()
```

### 第4步: _load_and_run_scenario() - 核心流程

```python
def _load_and_run_scenario(self, args, config):
    # =====================================================
    # 阶段1: 设置Agent
    # =====================================================
    print("> Setting up the agent")
    
    # 1.1 获取Agent入口点 (HybridAgent)
    agent_class_name = getattr(self.module_agent, 'get_entry_point')()
    
    # 1.2 实例化Agent
    # 调用 submission_agent.py 中的 HybridAgent(args.agent_config)
    self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
    
    # 1.3 验证传感器配置
    self.sensors = self.agent_instance.sensors()
    AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)
    
    # =====================================================
    # 阶段2: 加载世界
    # =====================================================
    print("> Loading the world")
    
    # 2.1 加载指定城镇地图
    self.world = self.client.load_world(config.town)
    
    # 2.2 设置同步模式，帧率20Hz
    settings.fixed_delta_seconds = 1.0 / 20.0  # 每帧50ms
    settings.synchronous_mode = True
    self.world.apply_settings(settings)
    
    # 2.3 生成ego车辆
    self._prepare_ego_vehicles(config.ego_vehicles, False)
    
    # 2.4 创建RouteScenario
    scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
    
    # =====================================================
    # 阶段3: 运行场景
    # =====================================================
    print("> Running the route")
    
    # 3.1 加载场景到ScenarioManager
    self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)
    
    # 3.2 运行场景 (核心循环)
    self.manager.run_scenario()
    
    # =====================================================
    # 阶段4: 清理
    # =====================================================
    print("> Stopping the route")
    
    self.manager.stop_scenario()
    self._register_statistics(config, args.checkpoint, entry_status, crash_message)
    scenario.remove_all_actors()
    self._cleanup()
```

---

## 三、ScenarioManager.run_scenario() - 仿真主循环

```python
def run_scenario(self):
    self._running = True
    
    while self._running:
        # 1. 获取传感器数据
        input_data = self._agent.sensor_interface.get_data()
        
        # 2. 获取当前时间戳
        timestamp = GameTime.get_time()
        
        # 3. 调用Agent的run_step (核心!)
        #    这会调用 submission_agent.py 中的 HybridAgent.run_step()
        control = self._agent.run_step(input_data, timestamp)
        
        # 4. 应用控制到ego车辆
        self.ego_vehicles[0].apply_control(control)
        
        # 5. 更新场景 (物理模拟、NPC行为等)
        self.scenario_tree.tick_once()
        
        # 6. 推进仿真一帧
        CarlaDataProvider.get_world().tick()
        
        # 7. 更新观察者视角
        spectator = CarlaDataProvider.get_world().get_spectator()
        ego_trans = self.ego_vehicles[0].get_transform()
        spectator.set_transform(...)
```

---

## 四、HybridAgent.run_step() - 每帧推理

```python
# submission_agent.py

@torch.inference_mode()
def run_step(self, input_data, timestamp):
    self.step += 1
    
    # 1. 处理传感器数据
    tick_data = self.tick(input_data)
    # - 获取RGB图像 (裁剪、拼接)
    # - 获取LiDAR点云 (体素化)
    # - 获取GPS、速度、罗盘
    # - 计算目标点
    
    # 2. 准备模型输入
    image = self.prepare_image(tick_data)        # RGB Tensor [1, 3, 160, 704]
    lidar_bev = self.prepare_lidar(tick_data)    # LiDAR Tensor [1, 2, 256, 256]
    target_point_image, target_point = self.prepare_goal_location(tick_data)
    velocity = torch.FloatTensor([tick_data['speed']]).reshape(1, 1)
    
    # 3. 模型推理 (集成N个模型)
    pred_wps = []
    for i in range(self.model_count):
        pred_wp, _ = self.nets[i].forward_ego(
            image, lidar_bev, target_point, target_point_image, velocity,
            save_path=SAVE_PATH, debug=self.config.debug
        )
        pred_wps.append(pred_wp)
    
    # 4. 集成预测 (取平均)
    self.pred_wp = torch.stack(pred_wps, dim=0).mean(dim=0)
    
    # 5. PID控制器计算转向、油门、刹车
    steer, throttle, brake = self.nets[0].control_pid(self.pred_wp, gt_velocity, is_stuck)
    
    # 6. 安全检查 (LiDAR检测前方障碍物)
    if self.use_lidar_safe_check and len(safety_box) > 0:
        control.brake = True
    
    # 7. 返回控制命令
    control = carla.VehicleControl()
    control.steer = float(steer)
    control.throttle = float(throttle)
    control.brake = float(brake)
    return control
```

---

## 五、数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CARLA 仿真器                                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ RGB相机 │  │ LiDAR   │  │ GPS/IMU │  │ 速度计  │  │ 场景/NPC│          │
│  │ (960x480)│ │ (点云)  │  │         │  │         │  │         │          │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │
└───────┼────────────┼────────────┼────────────┼────────────┼────────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            │
┌───────────────────────────────────────────────────────────┐│
│                    submission_agent.py                     ││
│  ┌─────────────────────────────────────────────────────┐  ││
│  │ tick() - 数据预处理                                   │  ││
│  │ - RGB裁剪: 2880x480 → 160x704                       │  ││
│  │ - LiDAR体素化: 点云 → 256x256网格                   │  ││
│  │ - 计算目标点                                          │  ││
│  └─────────────────────────────────────────────────────┘  ││
│                          │                                 ││
│                          ▼                                 ││
│  ┌─────────────────────────────────────────────────────┐  ││
│  │ forward_ego() - 模型推理                              │  ││
│  │                                                       │  ││
│  │   RGB ──→ ImageEncoder ──┐                           │  ││
│  │                          ├──→ Transformer ──→ BEV    │  ││
│  │   LiDAR ──→ LidarEncoder ┘       Fusion      Features│  ││
│  │                                     │                 │  ││
│  │                                     ▼                 │  ││
│  │                              ┌──────────────┐         │  ││
│  │                              │ 检测头        │         │  ││
│  │                              │ - 3D边界框    │         │  ││
│  │                              │ - 路径点      │         │  ││
│  │                              │ - BEV分割     │         │  ││
│  │                              └──────────────┘         │  ││
│  └─────────────────────────────────────────────────────┘  ││
│                          │                                 ││
│                          ▼                                 ││
│  ┌─────────────────────────────────────────────────────┐  ││
│  │ control_pid() - PID控制器                             │  ││
│  │ - 转向: PID控制                                       │  ││
│  │ - 油门/刹车: 纵向控制                                 │  ││
│  └─────────────────────────────────────────────────────┘  ││
└───────────────────────────────────────────────────────────┘│
        │                                                    │
        ▼                                                    │
┌─────────────────────────────────────────────────────────────┘
│  VehicleControl: {steer, throttle, brake}
│
└───────► CARLA ego车辆执行控制
```

---

## 六、关键时间点

| 阶段 | 时间 | 说明 |
|------|------|------|
| CARLA物理模拟 | 15-25ms | 物理引擎计算 |
| 传感器渲染 | 10-20ms | RGB/LiDAR渲染 |
| 数据预处理 | 2-5ms | 图像裁剪、点云体素化 |
| 模型推理 | 20-70ms | 神经网络前向传播 |
| 后处理+控制 | 1-5ms | PID控制、安全检查 |
| **总计** | **50-120ms** | 每帧总时间 |

---

## 七、输出文件

| 文件 | 说明 |
|------|------|
| `transfuser_longest6.json` | 评估结果JSON |
| `results/visualizations/*.png` | 可视化图像 |
| `results/parsed/results.csv` | 解析后的CSV报告 |