#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager_local import ScenarioManager
from leaderboard.scenarios.route_scenario_local import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper_local import  AgentWrapper, AgentError
from leaderboard.utils.statistics_manager_local import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer


sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer',
    'sensor.stitch_camera.rgb': 'carla_camera',  # for local World on Rails evaluation
    'sensor.camera.semantic_segmentation': 'carla_camera', # for datagen
    'sensor.camera.depth':      'carla_camera', # for datagen
}


class LeaderboardEvaluator(object):

    """
    这个类基本就是整个 leaderboard 本地评估流程的总控台。

    从外面看，它做的事情其实很直白：
    1. 连上 CARLA，准备好世界和 ScenarioManager；
    2. 动态加载用户自己的 agent；
    3. 按 routes / scenarios 配置，把每一条路线逐个拿出来跑；
    4. 每条路线结束后记分、落盘、清理现场；
    5. 全部跑完之后，再汇总成一份总成绩。

    所以后面读这个文件时，可以一直带着这条主线：
    `main -> run -> _load_and_run_scenario -> register statistics -> cleanup`
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        评估器初始化阶段先把“跑评测需要的基础设施”搭起来。

        这里还不会真正开始跑路线，主要是先把几样长期存在的东西准备好：
        - 连到 CARLA server；
        - 把待测 agent 的 Python 模块动态 import 进来；
        - 创建 ScenarioManager，后面每条路线都交给它来驱动；
        - 准备 watchdog，防止 agent 初始化时卡死。
        """
        self.statistics_manager = statistics_manager
        self.sensors = None
        self.sensor_icons = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # 先把 CARLA client 建起来。后面加载世界、生成车辆、拿 traffic manager
        # 都要从这个 client 往 simulator 发请求。
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        self.world = self.client.load_world('Town01')
        self.traffic_manager = self.client.get_trafficmanager(int(args.trafficManagerPort))

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # 用户传进来的是一个 agent 的 py 文件路径，这里把它当成模块动态加载。
        # 真正实例化 agent 的动作放到每条路线开始前做，这样失败了只影响当前 route。
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # ScenarioManager 是真正“驱动仿真往前跑”的角色。
        # evaluator 更像总导演，manager 更像现场执行。
        self.manager = ScenarioManager(args.timeout, args.debug > 1)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # agent watchdog 单独盯 agent 初始化这段，避免模型加载或配置出问题时一直挂着。
        self._agent_watchdog = Watchdog(int(float(args.timeout)))
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """
        收到 Ctrl+C 之类的中断时，不是简单粗暴地直接退出，
        而是优先判断当前是不是卡在 agent 初始化阶段，
        再决定抛超时还是把信号转交给 ScenarioManager 去收尾。
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took too long to setup")
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        对象销毁前再兜底清一次，尽量别把车辆、manager、world 残留在进程里。
        """

        self._cleanup()
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self):
        """
        每条路线跑完后，这里负责把现场收干净。

        这一步很重要，因为 leaderboard 是一条 route 接一条 route 连着跑的。
        如果上一条路线留下同步模式、ego 车、agent 传感器或者 manager 状态，
        下一条路线很容易出现看起来很玄学、其实只是没清干净的问题。
        """

        # 如果仿真还停在同步模式里，先把世界切回异步。
        # 不然 CARLA server 可能一直等 tick，后面的清理和下一次加载都会别扭。
        if self.manager and self.manager.get_running_status() \
                and hasattr(self, 'world') and self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(False)

        if self.manager:
            self.manager.cleanup()

        CarlaDataProvider.cleanup()

        # evaluator 自己维护了一份 ego 列表，这里逐个销毁，保证 route 之间不串车。
        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

        if hasattr(self, 'statistics_manager') and self.statistics_manager:
            self.statistics_manager.scenario = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        把当前 route 需要的 ego 车准备出来。

        这里有两种模式：
        - 直接生成：最常见，按配置在世界里 spawn 新车；
        - 等待现成车辆：有些模式下 ego 已经在世界里了，这时就去 world 里把它们认出来。
        """

        if not wait_for_ego_vehicles:
            # 常规 leaderboard 流程走这里：按 route 配置直接创建 ego 车。
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaDataProvider.request_new_actor(vehicle.model,
                                                                             vehicle.transform,
                                                                             vehicle.rolename,
                                                                             color=vehicle.color,
                                                                             vehicle_category=vehicle.category))

        else:
            # 这个分支更像“接管已有车辆”。
            # 会一直等到世界里真的出现了对应 role_name 的车，再把它们的位置同步到配置里的起点。
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # 最后主动 tick 一下，让刚创建/更新过的车辆状态真正落到仿真里。
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        为当前 route 切到对应 town，并把世界调成 leaderboard 期望的运行方式。

        这一段做的事情可以理解成“把舞台先搭好”：
        - 加载正确地图；
        - 切同步模式和固定步长；
        - 把 client / world / traffic manager 注册给 CarlaDataProvider；
        - 等世界至少稳定 tick 一次，再继续后面的 route 构建。
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        self.world.reset_all_traffic_lights()

        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(int(args.trafficManagerPort))

        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(int(args.trafficManagerSeed))

        # 世界刚 load 完时，很多依赖 world state 的对象还没完全稳定。
        # 这里先等一拍，后面 route / actor / map 的读取会更稳。
        if CarlaDataProvider.is_sync_mode():
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            "This scenario requires to use map {}".format(town))

    def _register_statistics(self, config, checkpoint, entry_status, crash_message=""):
        """
        把这一条 route 的结果算出来并写进 checkpoint。

        注意这里不是简单写个成功失败，而是会把 route 完成度、处罚项、最终 composed score
        一起算好，所以它基本就是“这一条路线的成绩落盘口”。
        """
        current_stats_record = self.statistics_manager.compute_route_statistics(
            config,
            self.manager.scenario_duration_system,
            self.manager.scenario_duration_game,
            crash_message
        )

        print("\033[1m> Registering the route statistics\033[0m")
        self.statistics_manager.save_record(current_stats_record, config.index, checkpoint)
        self.statistics_manager.save_entry_status(entry_status, False, checkpoint)

    def _load_and_run_scenario(self, args, config):
        """
        这是单条 route 的完整执行流程，也是整个文件最核心的函数。

        一条 route 跑下来基本就四步：
        1. 先把 agent 实例化好，并检查它声明的传感器是否合法；
        2. 再加载 town、ego 车和 RouteScenario，把仿真场景真正搭起来；
        3. 然后交给 ScenarioManager 一帧一帧地跑，直到完成、agent 出错或仿真出错；
        4. 最后无论结果好坏，都尽量停场景、记成绩、清理资源。

        这里异常处理分得比较细，是因为 leaderboard 希望区分：
        - 只是当前 route 失败了，那就记失败后继续下一条；
        - 整个仿真环境炸了，那就直接退出，避免后面的结果不可信。
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        # 先把这条 route 在统计器里占一个坑位。后面不管成败，结果都会回写到这里。
        self.statistics_manager.set_route(config.name, config.index)
        if int(os.environ['DATAGEN'])==1:
            CarlaDataProvider._rng = random.RandomState(config.index)

        # 第一阶段：起 agent。
        # 这里故意把 agent 初始化放在 world/scenario 之前，这样模型、配置、依赖有问题时，
        # 可以尽早失败，不用先把整套仿真现场都搭起来。
        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            if int(os.environ['DATAGEN'])==1:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config, config.index)
            else:
                self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config)
            config.agent = self.agent_instance

            # 传感器配置只在第一次 route 时检查并保存一次。
            # 因为同一个 agent 后面几条 route 理论上不会突然换一套传感器。
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons, args.checkpoint)

            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            # 传感器不合法是“提交本身就不符合规则”，所以直接按 Rejected 处理并退出。
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            # agent 起不来，通常说明模型权重、配置、依赖或构造函数有问题。
            # 这类错误只记当前 route，流程上允许继续尝试下一条。
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent couldn't be set up"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")

        # 第二阶段：把这条 route 对应的世界和场景搭起来。
        # 到这里才真正开始跟当前 town、ego 车、scenario tree 发生关系。
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(scenario.scenario)

            # 如果太阳已经下去了，给 ego 开灯。
            # 这不是评分逻辑，而是尽量让运行环境跟夜间驾驶的设定一致。
            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            # record 打开后会把这一条 route 的 CARLA 回放文件落下来，方便事后复盘。
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(scenario, self.agent_instance, config.repetition_index)

        except Exception as e:
            # 场景都没能成功加载，后面就没法相信还能继续跑，所以这里按仿真崩溃处理并退出。
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        # 第三阶段：真正开跑。
        # 一旦进入 manager.run_scenario()，控制权基本就交给 ScenarioManager 了。
        # 它会循环做：读传感器 -> 调 agent -> 下控制 -> tick scenario tree -> tick world。
        try:
            self.manager.run_scenario()

        except AgentError as e:
            # 这里单独抓 AgentError，是为了把“agent 自己跑挂了”和“CARLA/场景挂了”分开记。
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Agent crashed"

        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

        # 第四阶段：收尾。
        # 不管 route 是成功跑完、agent 崩了，还是中途异常，只要还能收尾，
        # 就尽量把 stop / statistics / recorder / actor cleanup 做完整。
        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.checkpoint, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            # RouteScenario 自己持有的演员也要清掉，不然下一条 route 会被上一条的残留污染。
            scenario.remove_all_actors()

            self._cleanup()

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()

            crash_message = "Simulation crashed"

        if crash_message == "Simulation crashed":
            sys.exit(-1)

    def run(self, args):
        """
        这里是“多条路线批量评估”的主循环。

        `RouteIndexer` 会把 routes/scenarios/repetitions 展开成一个待执行队列，
        然后 evaluator 就不断地：
        - 取下一条 config；
        - 调 `_load_and_run_scenario()` 跑掉它；
        - 把当前进度写进 checkpoint。

        所以如果中途停了，`--resume` 本质上就是从这个队列的中间位置继续往后跑。
        """
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions)

        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            route_indexer.save_state(args.checkpoint)

        while route_indexer.peek():
            # 这里拿到的 config 就是“一条具体 route 的一次具体 repetition”。
            config = route_indexer.next()

            # 真正的单条路线执行都封装在这里面。
            self._load_and_run_scenario(args, config)

            # 每跑完一条就存一次进度，避免长实验中途挂了之后前面的结果白跑。
            route_indexer.save_state(args.checkpoint)

        # 全部路线结束后，再做一次总汇总。
        # 这时候产出的就是 leaderboard 最终看见的那组平均分和违规率。
        print("\033[1m> Registering the global statistics\033[0m")
        global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
        StatisticsManager.save_global_record(global_stats_record, self.sensor_icons, route_indexer.total, args.checkpoint)


def main():
    # `main` 自己不做复杂逻辑，职责很单纯：
    # 解析命令行参数 -> 构造统计器和评估器 -> 启动整个评测流程。
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # 这一组是跟 CARLA server 和运行时行为有关的通用参数。
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='0',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="60.0",
                        help='Set the CARLA client timeout value in seconds')

    # 这一组决定“要跑哪些路线、搭哪些 scenario、每条重复几次”。
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # 这一组决定“拿哪个 agent 来测、它的配置是什么、成绩写到哪里”。
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        # 整个 leaderboard 的实际入口就在这里。
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
