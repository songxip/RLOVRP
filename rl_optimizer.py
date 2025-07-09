import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow.keras as keras
from keras.api._v2 import keras
from keras import layers,Model
import pickle
import os
import random
import copy
from utils import *
import time
import sys # Import sys for RLEnhancedScheduler output redirection
from keras.callbacks import TensorBoard
import datetime


# --- Constants for RLOptimizer ---
TIME_WINDOW_OPTIMAL = 10 # 分钟
TIME_WINDOW_ACCEPTABLE_EXTENSION = 20 # 分钟
PENALTY_EARLY = 0.7
PENALTY_LATE = 0.8
AVG_SPEED = 60.0 # km/h
CHARGING_POWER = 80.0 # kW
MAX_SERVICE_CAPACITY = 500.0 # kWh
MAX_RANGE = 600.0 # km

# --- Global constants for normalization (Derived from Solomon C101 typically) ---
GLOBAL_MAX_MAP_DIM = 100.0  # Max X or Y coordinate
GLOBAL_MAX_DEMAND = 70.0   # Max demand (adjust if needed)
GLOBAL_MAX_SOLOMON_TIME = 1236.0 # Max due time in Solomon C101 in minutes
GLOBAL_MAX_SIM_TIME_HOURS = GLOBAL_MAX_SOLOMON_TIME / 60.0 # Max sim time in hours
GLOBAL_MAX_SERVICE_CAPACITY = 500.0
GLOBAL_MAX_RANGE = 600.0
GLOBAL_AVG_WINDOW_WIDTH_HOURS = 2.0 # Rough average window width in hours

class RLOptimizer:
    """强化学习优化器类"""
    
    # --- 核心奖励/惩罚常量 (移到类级别) ---
    REWARD_BASE_SUCCESS = 40.0         # 基础成功奖励
    REWARD_OPTIMAL_WINDOW_BONUS = 30.0 # 最优窗额外奖励
    PENALTY_TIME_DEVIATION_FACTOR = 3.0 # 时间偏差惩罚系数 (惩罚 = 系数 * 偏差小时)
    PENALTY_TRAVEL_TIME_FACTOR = 0.8   # 行驶时间惩罚系数 (惩罚 = 系数 * 行驶小时)
    PENALTY_WAIT_TIME_FACTOR = 0.2     # 等待时间惩罚系数 (如果车辆需等待请求ready_time)
    PENALTY_HIGH_CAPACITY_USAGE_FACTOR = 0.0 # 暂时禁用
    PENALTY_LOW_REMAINING_RANGE_FACTOR = 0.0 # 暂时禁用
    # REWARD_FAILURE = -75.0            # 分配失败或窗外到达的惩罚 (保留，但数值可能需要调整)
    # 调整失败惩罚，使其与成本尺度相关
    # 增大失败惩罚，使其显著高于一般任务的成本
    PENALTY_FAILURE_COST_EQUIVALENT = 100.0 # 失败相当于多少小时的成本惩罚
    REWARD_FAILURE = -PENALTY_FAILURE_COST_EQUIVALENT # 保持定义，用于外部访问

    # --- 成本因子 (用于计算奖励) ---
    COST_FACTOR_TRAVEL = 1.0 # 行驶时间成本系数
    COST_FACTOR_CHARGE = 1.0 # 充电时间成本系数
    COST_FACTOR_PENALTY = 1.0 # 惩罚时间成本系数
    # --------------------------------

    NUM_RULES = 6 # <--- 定义规则数量 (增加到 6)

    def __init__(self, state_dim=10, learning_rate=0.0002, gamma=0.95, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01,mode = "train"):
        self.state_dim = state_dim  # 状态维度
        self.action_dim = self.NUM_RULES  # <--- 动作维度 = 规则数量
        self.gamma = gamma  # 折扣因子
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.epsilon_min = epsilon_min  
        self.learning_rate = learning_rate  # 学习率
        self.mode = mode
        if self.mode == "single" or self.mode == "compare":
            self.epsilon = 0
        else :
            self.epsilon = 1.0
        
        # 计数器 (移到前面)
        self.train_counter = 0
        
        # 创建深度Q网络
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # 现在调用时 train_counter 已存在
        
        # 经验回放缓存
        self.memory = []
        self.max_memory_size = 50000 # Increased memory size
        
        # 批处理大小
        self.batch_size = 64 # Increased batch size


    def log_reward(self, reward, step):
        """把当前reward写到TensorBoard"""
        with self.writer.as_default():
            tf.summary.scalar('reward', reward, step=step)
            self.writer.flush()

    def log_epsilon(self, epsilon, step):
        """把当前epsilon写到TensorBoard"""
        with self.writer.as_default():
            tf.summary.scalar('epsilon', epsilon, step=step)
            self.writer.flush()


    
    def _build_model(self):
        """构建深度Q网络模型"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_dim,)), # Increased layer size
            layers.Dropout(0.2), # Added dropout
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2), # Added dropout
            layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss=tf.keras.losses.Huber()) # Changed loss to Huber
        return model
    
    def update_target_model(self):
        """更新目标网络权重"""
        if self.train_counter % 100 == 0: # Update target less frequently
            self.target_model.set_weights(self.model.get_weights())
    
    def encode_state(self, vehicles, request, current_time):
        """
        将当前状态编码为向量表示 (使用全局常量归一化)
        """
        state = []

        # --- 使用全局常量进行归一化 ---
        MAX_MAP_DIM = GLOBAL_MAX_MAP_DIM
        MAX_CHARGE = GLOBAL_MAX_DEMAND
        MAX_SIM_TIME = GLOBAL_MAX_SIM_TIME_HOURS
        MAX_CAPACITY = GLOBAL_MAX_SERVICE_CAPACITY
        MAX_RANGE = GLOBAL_MAX_RANGE
        AVG_WINDOW_WIDTH = GLOBAL_AVG_WINDOW_WIDTH_HOURS
        # ---------------------------------
        
        # 编码请求信息
        state.append(request.location[0] / MAX_MAP_DIM)
        state.append(request.location[1] / MAX_MAP_DIM)
        state.append(request.charge_amount / MAX_CHARGE)
        # 时间窗信息相对于当前时间 和 窗口宽度 (使用小时)
        state.append(max(0, request.li - current_time) / max(AVG_WINDOW_WIDTH, 0.1)) # 避免除零
        state.append(max(0, request.sli - current_time) / max(AVG_WINDOW_WIDTH, 0.1)) # 避免除零
        state.append((request.li - request.ei) / max(AVG_WINDOW_WIDTH, 0.1)) # 避免除零, 增加最优时间窗宽度信息
        
        # 当前时间信息
        state.append(current_time / MAX_SIM_TIME)
        
        # 对每辆车选取最重要的状态信息
        max_vehicle_range = MAX_RANGE # Assuming all vehicles have same max range for normalization
        max_vehicle_capacity = MAX_CAPACITY # Assuming all vehicles have same max capacity

        vehicle_features = []
        for vehicle in vehicles:
            # 车辆当前位置与请求位置的归一化距离
            distance = calculate_distance(vehicle.current_location, request.location)
            norm_dist = distance / (MAX_MAP_DIM * 1.414)
            vehicle_features.append(norm_dist)
            
            # 车辆是否空闲
            vehicle_features.append(1.0 if vehicle.is_idle else 0.0)
            
            # 归一化剩余服务电量
            vehicle_features.append(vehicle.remaining_service_capacity / max(max_vehicle_capacity, 1e-6)) # Avoid zero division
            
            # 归一化剩余续航里程
            vehicle_features.append(vehicle.remaining_travel_distance / max(max_vehicle_range, 1e-6)) # Avoid zero division
            
            # 车辆下一次可用时间（相对于当前时间）
            relative_available_time = max(0, vehicle.available_time - current_time)
            vehicle_features.append(relative_available_time / max(MAX_SIM_TIME, 1e-6)) # Avoid zero division

            # (新增) 车辆能否物理上服务此请求 (不考虑时间窗)
            can_serve_physically = (vehicle.remaining_travel_distance >= distance + calculate_distance(request.location, vehicle.depot_location) and
                                    vehicle.remaining_service_capacity >= request.charge_amount)
            vehicle_features.append(1.0 if can_serve_physically else 0.0)


        # --- 状态向量填充/截断 ---
        # Base state: request info (6) + time (1) = 7
        # Per vehicle features = 6
        # Total expected state size = 7 + num_vehicles * 6
        # Ensure self.state_dim in __init__ matches this calculation based on MAX_VEHICLES
        state.extend(vehicle_features)
        # --------------------------

        current_len = len(state)
        if current_len < self.state_dim:
            state.extend([0.0] * (self.state_dim - current_len)) # Pad with zeros
        elif current_len > self.state_dim:
            state = state[:self.state_dim] # Truncate
            
        return np.array(state)
    
    def calculate_reward(self, request, assigned_vehicle, actual_arrival_time, assignment_successful, is_optimal, base_scheduler_c1, base_scheduler_c2, current_time):
        """
        计算奖励函数 (使用更新后的常量和逻辑)

        Args:
            request: The request being processed.
            assigned_vehicle: The vehicle assigned (or None if failed).
            actual_arrival_time: The actual arrival time (or None if failed/rejected).
            assignment_successful: Boolean indicating if base_scheduler.assign_request succeeded.
            is_optimal: Boolean indicating if arrival was within the optimal window [ei, li].
            base_scheduler_c1: Early penalty coefficient from the scheduler.
            base_scheduler_c2: Late penalty coefficient from the scheduler.
            current_time: Current simulation time.
        """
        if not assignment_successful or actual_arrival_time is None or actual_arrival_time > request.sli:
            # 失败就返回失败成本
            # return -self.PENALTY_FAILURE_COST_EQUIVALENT # 失败惩罚
            return self.REWARD_FAILURE # 使用统一的失败奖励

        # --- 计算成功分配的总成本 ---
        total_cost = 0.0

        # 1. 惩罚时间成本
        time_penalty_hours = request.calculate_arrival_penalty(actual_arrival_time, base_scheduler_c1, base_scheduler_c2)
        if time_penalty_hours is None:
            time_penalty_hours = 0
        total_cost += self.COST_FACTOR_PENALTY * time_penalty_hours

        # 2. 行驶时间成本 + 3. 充电时间成本
        if assigned_vehicle and request:
            try:
                # 行驶时间
                prev_loc = getattr(assigned_vehicle, 'previous_location', assigned_vehicle.depot_location)
                if prev_loc is None:
                    prev_loc = assigned_vehicle.depot_location
                travel_time = calculate_travel_time(prev_loc, request.location, assigned_vehicle.speed)
                total_cost += self.COST_FACTOR_TRAVEL * travel_time

                # 充电时间
                charge_time = request.charge_amount / assigned_vehicle.charging_power
                total_cost += self.COST_FACTOR_CHARGE * charge_time

                # (可选) 可以保留或移除等待时间惩罚，这里暂时移除，因为目标是总成本
                # wait_time = max(0, request.ei - actual_arrival_time)
                # total_cost += self.COST_FACTOR_WAIT * wait_time # 需要定义 COST_FACTOR_WAIT

            except AttributeError as e:
                print(f"[Warning] calculate_reward: AttributeError accessing vehicle properties: {e}. Cost calculation might be inaccurate.")
            except Exception as e:
                print(f"[Warning] calculate_reward: Error during cost calculation: {e}. Cost calculation might be inaccurate.")
        else:
            print("[Warning] calculate_reward: assigned_vehicle or request is None. Cannot calculate travel/charge costs.")

        # 奖励是负的总成本
        reward = -total_cost
        # print(f"[Debug Reward OK] Req {request.id} Veh {assigned_vehicle.id}: Arr={actual_arrival_time:.2f} Pen={time_penalty_hours:.2f} Trav={travel_time:.2f} Chg={charge_time:.2f} Cost={total_cost:.2f} Reward={reward:.2f}")

        return reward
    
    def select_action(self, state, available_actions):
        """选择动作（选择哪个规则）"""
        if self.mode == "single" or self.mode ==  "comapare":
            self.epsilon = 0 
        if not available_actions:
            # print("[Debug] select_action: No available actions.")
            return None # Indicate no action can be taken
        
        print(f"epsilon: {self.epsilon}")
        # Ensure available actions are within the valid range
        valid_available_actions = [a for a in available_actions if 0 <= a < self.action_dim]
        if not valid_available_actions:
            # print(f"[Warning] select_action: No valid available actions from {available_actions}. Using random.")
            # This case should ideally not happen if available_rules logic is correct
            return None # Or maybe random.choice(available_actions) if sure they exist but map wrong?

        if len(valid_available_actions) == 1:
            return valid_available_actions[0]
        
        if np.random.rand() < self.epsilon:
           # print(f"[Debug] Exploring: Choosing random from {valid_available_actions}")
            return np.random.choice(valid_available_actions)
        else:
            # q_values = self.model.predict(state.reshape(1, -1))[0]

            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            q_values = self.model(state_tensor, training=(self.mode != "single")).numpy()[0]

            # print(f"[Debug] Exploiting: Q-values: {q_values}, Available: {valid_available_actions}")
            # Filter Q-values for available actions only
            valid_q_values = {action: q_values[action] for action in valid_available_actions}

            if not valid_q_values:
                # This should not happen if valid_available_actions is not empty
                print(f"[Error] select_action: No valid Q-values found for available actions {valid_available_actions}. Choosing random.")
                return np.random.choice(valid_available_actions)

            best_action = max(valid_q_values, key=valid_q_values.get)
            # print(f"[Debug] Exploiting: Chose action {best_action} with Q-value {valid_q_values[best_action]}")
            return best_action
    
    def remember(self, state, action, reward, next_state, done, available_next_actions):
        """存储经验到回放缓存 (带检查)"""
        if not (0 <= action < self.action_dim):
            print(f"[Error] RLOptimizer.remember: Invalid action index: {action}")
            return
        
        if len(self.memory) >= self.max_memory_size:
            # Prioritized Experience Replay could be added here
            self.memory.pop(random.randrange(len(self.memory))) # Random eviction

        # Validate available_next_actions before storing
        valid_next_actions = [a for a in available_next_actions if 0 <= a < self.action_dim]
        # if len(valid_next_actions) != len(available_next_actions):
        #      print(f"[Warning] RLOptimizer.remember: Invalid next actions removed from {available_next_actions}")

        self.memory.append((np.array(state), action, reward, np.array(next_state), done, valid_next_actions))

    def replay(self):
        """从经验回放缓存中抽样批次进行训练 (DQN with Target Network)"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        available_next_actions_list = [transition[5] for transition in minibatch]

        # Predict Q-values for current states using main model
        current_q_values = self.model.predict(states)
        # Predict Q-values for next states using target model
        next_q_values_target = self.target_model.predict(next_states)

        # Prepare target Q-values for batch update
        target_q_values = np.copy(current_q_values)

        for i in range(self.batch_size):
            action = actions[i]
            reward = rewards[i]
            done = dones[i]
            available_next_actions = available_next_actions_list[i]
            
            if not (0 <= action < self.action_dim):
                print(f"[Error] RLOptimizer.replay: Invalid action index {action} in minibatch. Skipping sample.")
                continue # Skip this sample

            if done:
                target = reward
            else:
                if not available_next_actions:
                    # If no actions available in next state, Q value is 0
                    max_next_q = 0.0
                    # print(f"[Debug] Replay: No available next actions for sample {i}")
                else:
                    # Get Q-values from target network for available next actions
                    valid_next_q = [next_q_values_target[i, a] for a in available_next_actions]
                    if not valid_next_q:
                        # print(f"[Warning] Replay: No valid next Q-values found for sample {i} despite available actions {available_next_actions}. Setting max_next_q to 0.")
                        max_next_q = 0.0
                    else:
                        max_next_q = np.max(valid_next_q)
                        # print(f"[Debug] Replay: Sample {i}, Available Next: {available_next_actions}, Next Qs: {valid_next_q}, MaxQ: {max_next_q}")

                
                target = reward + self.gamma * max_next_q
            
            target_q_values[i, action] = target
        
        # Train the main model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)
        self.train_counter += 1

        # # Update Epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        # Update Target Network (less frequently)
        self.update_target_model()
    
    def save_model(self, filepath):
        """保存模型权重和优化器状态"""
        print(f"尝试保存 RL 模型到: {filepath}") 
        try:
            self.model.save_weights(filepath)
            # Consider saving epsilon value too if needed for resuming training state
            print(f"RL 模型权重已成功保存到 {filepath}")
        except Exception as e:
            print(f"错误: 保存 RL 模型到 {filepath} 失败: {e}")
    
    def load_model(self, filepath):
        """加载模型权重"""
        if os.path.exists(filepath):
            print(f"尝试从 {filepath} 加载 RL 模型权重...")
            try:
                self.model.load_weights(filepath)
                self.update_target_model() # Ensure target network matches loaded weights
                print(f"RL 模型权重已成功从 {filepath} 加载。")
                # Consider loading epsilon value if saved
            except Exception as e:
                print(f"警告: 加载 RL 模型权重从 {filepath} 失败: {e}。将使用初始化的模型。")
        else:
            print(f"警告: 找不到 RL 模型文件 {filepath}。将使用新初始化的模型。")

# --- RLEnhancedScheduler Class Definition --- 

# Assuming OnlineScheduler is defined elsewhere (e.g., scheduler.py)
# from scheduler import OnlineScheduler 

class RLEnhancedScheduler:
    """强化学习增强的调度器 (基于规则选择)"""
    
    def __init__(self, base_scheduler, max_num_vehicles, use_rl=True,mode = "train"):
        """初始化
        Args:
            base_scheduler: 基础启发式调度器实例。
            max_num_vehicles: 系统支持的最大车辆数 (用于确定状态维度)。
            use_rl: 是否启用 RL。
        """
        self.base_scheduler = base_scheduler
        self.use_rl = use_rl
        self.rl_decision_count = 0
        self.rl_improvements_count = 0 # Track RL improvements (optional)
        self.mode = mode

        # Calculate state dimension based on max_num_vehicles
        # State: request (6) + time (1) + max_vehicles * features_per_vehicle (6)
        state_dim = 7 + max_num_vehicles * 6
        print(f"Initializing RLOptimizer with state_dim={state_dim} (for max {max_num_vehicles} vehicles)")
        self.rl_optimizer = RLOptimizer(state_dim=state_dim,mode=mode)
    
    def get_idle_vehicles(self):
        """获取空闲车辆"""
        return self.base_scheduler.get_idle_vehicles()
    
    def get_busy_vehicles(self):
        """获取忙碌车辆"""
        return self.base_scheduler.get_busy_vehicles()

    # --- 新增: 规则实现辅助函数 ---

    def _find_vehicle_rule_0_min_cost(self, request, current_time):
        """规则 0: 查找最小总成本增量的车辆 (近似)
           近似：优先空闲车的最早到达，其次忙碌车的最优窗最早到达。
           返回 (best_vehicle, arrival_time) 或 (None, None)
        """
        best_vehicle = None
        best_arrival_time = float('inf')
        min_cost_increase = float('inf') # 追踪成本增量

        candidate_vehicles_info = [] # (vehicle, arrival_time, cost_increase)

        # 1. 评估空闲车辆
        for vehicle in self.get_idle_vehicles():
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    # 近似成本增量：行驶时间 + 惩罚时间
                    travel_time = calculate_travel_time(vehicle.current_location, request.location, vehicle.speed)
                    penalty = request.calculate_arrival_penalty(arrival_time, self.base_scheduler.c1, self.base_scheduler.c2)
                    cost_increase = travel_time + penalty
                    candidate_vehicles_info.append((vehicle, arrival_time, cost_increase))

        # 2. 评估忙碌车辆
        for vehicle in self.get_busy_vehicles():
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    # 近似成本增量：行驶时间 + 惩罚时间 (从上一站)
                    # 假设车辆从完成上一个任务的地点出发
                    # 修正: vehicle.route 存储的是 (location, arrival, departure) 元组
                    last_loc = vehicle.route[-1][0] if vehicle.route else vehicle.depot_location
                    travel_time = calculate_travel_time(last_loc, request.location, vehicle.speed)
                    penalty = request.calculate_arrival_penalty(arrival_time, self.base_scheduler.c1, self.base_scheduler.c2)
                    cost_increase = travel_time + penalty
                    candidate_vehicles_info.append((vehicle, arrival_time, cost_increase))

        # 选择成本增量最小的
        if candidate_vehicles_info:
            # 按成本增量排序，然后按到达时间排序（成本相同时选早到的）
            candidate_vehicles_info.sort(key=lambda x: (x[2], x[1]))
            best_vehicle, best_arrival_time, _ = candidate_vehicles_info[0]
            return best_vehicle, best_arrival_time

        return None, None

    def _find_vehicle_rule_1_earliest_finish_idle(self, request, current_time):
        """规则 1: 查找最早完成服务的空闲车辆
        返回 (best_vehicle, arrival_time) 或 (None, None)
        """
        best_vehicle = None
        earliest_finish_time = float('inf')
        best_arrival_time = None

        for vehicle in self.get_idle_vehicles():
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    # 计算完成时间
                    charge_time = calculate_charging_time(request.charge_amount,power=40)
                    wait_time = max(0, request.ei - arrival_time) # 使用 ei 替代 ready_time
                    finish_time = arrival_time + wait_time + charge_time
                    if finish_time < earliest_finish_time:
                        earliest_finish_time = finish_time
                        best_vehicle = vehicle
                        best_arrival_time = arrival_time

        return best_vehicle, best_arrival_time

    def _find_vehicle_rule_2_nearest_idle(self, request, current_time):
        """规则 2: 查找最近的空闲车辆
           返回 (best_vehicle, arrival_time) 或 (None, None)
        """
        best_vehicle = None
        min_distance = float('inf')
        best_arrival_time = None

        for vehicle in self.get_idle_vehicles():
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    distance = calculate_distance(vehicle.current_location, request.location)
                    if distance < min_distance:
                        min_distance = distance
                        best_vehicle = vehicle
                        best_arrival_time = arrival_time

        return best_vehicle, best_arrival_time

    def _find_vehicle_rule_3_min_penalty(self, request, current_time):
        """规则 3: 查找惩罚时间最小的车辆 (优先最优窗)
           返回 (best_vehicle, arrival_time) 或 (None, None)
        """
        best_vehicle = None
        min_penalty = float('inf')
        best_arrival_time = None
        in_optimal = False # 是否找到最优窗内的

        candidate_vehicles_info = [] # (vehicle, arrival_time, penalty, is_optimal)
        for vehicle in self.base_scheduler.vehicles:
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    penalty = request.calculate_arrival_penalty(arrival_time, self.base_scheduler.c1, self.base_scheduler.c2)
                    is_opt = (request.ei <= arrival_time <= request.li)
                    candidate_vehicles_info.append((vehicle, arrival_time, penalty, is_opt))

        if candidate_vehicles_info:
            # 排序: 优先最优窗(is_opt=True, penalty=0), 然后按惩罚升序, 最后按到达时间升序
            candidate_vehicles_info.sort(key=lambda x: (not x[3], x[2], x[1]))
            best_vehicle, best_arrival_time, _, _ = candidate_vehicles_info[0]
            return best_vehicle, best_arrival_time

        return None, None

    def _find_vehicle_rule_4_max_slack(self, request, current_time):
        """规则 4: 查找服务完后剩余时间 (slack time) 最长的车辆（舍弃）
           Slack = sli - finish_time
           返回 (best_vehicle, arrival_time) 或 (None, None)
        """
        best_vehicle = None
        max_slack = -float('inf')
        best_arrival_time = None

        for vehicle in self.base_scheduler.vehicles:
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    charge_time = request.charge_amount / vehicle.charging_power
                    wait_time = max(0, request.ei - arrival_time) # 使用 ei 替代 ready_time
                    finish_time = arrival_time + wait_time + charge_time
                    slack = request.sli - finish_time
                    if slack > max_slack:
                        max_slack = slack
                        best_vehicle = vehicle
                        best_arrival_time = arrival_time

        return best_vehicle, best_arrival_time

    def _find_vehicle_rule_5_earliest_finish_any(self, request, current_time):
        """规则 5: 查找最早完成服务的所有车辆 (包括忙碌)
           返回 (best_vehicle, arrival_time) 或 (None, None)6
        """
        best_vehicle = None
        earliest_finish_time = float('inf')
        best_arrival_time = None

        for vehicle in self.base_scheduler.vehicles:
            if vehicle.can_serve(request, current_time):
                arrival_time = vehicle.estimate_arrival_time(request.location, current_time)
                if request.sei <= arrival_time <= request.sli:
                    charge_time = request.charge_amount / vehicle.charging_power
                    wait_time = max(0, request.ei - arrival_time) # 使用 ei 替代 ready_time
                    finish_time = arrival_time + wait_time + charge_time
                    if finish_time < earliest_finish_time:
                        earliest_finish_time = finish_time
                        best_vehicle = vehicle
                        best_arrival_time = arrival_time

        return best_vehicle, best_arrival_time

    # --- 结束: 新增规则实现辅助函数 ---

    def _get_available_rules(self, request, current_time):
        """Helper to determine which rules are currently feasible.
           Checks if at least one vehicle can satisfy the constraints for each rule.
           Returns a list of available rule indices.
        """
        available_rule_indices = []

        # 检查每个规则是否至少有一个可行车辆
        rule_functions = [
            self._find_vehicle_rule_0_min_cost,        # 规则 0
            self._find_vehicle_rule_1_earliest_finish_idle, # 规则 1
            self._find_vehicle_rule_2_nearest_idle,        # 规则 2
            self._find_vehicle_rule_3_min_penalty,       # 规则 3
            self._find_vehicle_rule_4_max_slack,         # 规则 4
            self._find_vehicle_rule_5_earliest_finish_any # 规则 5
        ]

        for i, rule_func in enumerate(rule_functions):
            best_vehicle, _ = rule_func(request, current_time)
            if best_vehicle is not None:
                available_rule_indices.append(i)

        # print(f"[Debug Req {request.id}] Available rules: {available_rule_indices}") # Optional debug print
        return available_rule_indices

    def _execute_rule(self, rule_index, request, current_time):
        """根据规则索引执行对应的查找函数"""
        rule_functions = [
            self._find_vehicle_rule_0_min_cost,
            self._find_vehicle_rule_1_earliest_finish_idle,
            self._find_vehicle_rule_2_nearest_idle,
            self._find_vehicle_rule_3_min_penalty,
            self._find_vehicle_rule_4_max_slack,
            self._find_vehicle_rule_5_earliest_finish_any
        ]
        if 0 <= rule_index < len(rule_functions):
            # print(f"[Debug Req {request.id}] Executing rule {rule_index}") # Optional debug print
            return rule_functions[rule_index](request, current_time)
        else:
            print(f"[Error Req {request.id}] Invalid rule index: {rule_index}")
            return None, None # Invalid index

    def process_request(self, request, is_last_request=False):
        """处理单个充电请求（基于规则选择的RL增强版）"""
        current_time = request.request_time # Use the correct attribute name
        self.base_scheduler.update_system_time(current_time) # Update base scheduler time too
        
        # 记录请求到达事件
        self.base_scheduler.record_snapshot(f"Request {request.id} arrives")

        # --- 1. 获取当前状态 ---
        vehicles = self.base_scheduler.vehicles # Get current vehicle state
        state = self.rl_optimizer.encode_state(vehicles, request, current_time)

        # --- 2. 获取可用规则 ---
        # (需要确认 _get_available_rules 的逻辑是否总是返回至少一个规则)
        available_rule_indices = self._get_available_rules(request, current_time)
        if not available_rule_indices:
            print(f"[Warning Req {request.id}] No available rules found. Rejecting request.")
            self.base_scheduler.rejected_requests.append(request)
            request.is_served = False
            # 记录请求被拒绝事件
            self.base_scheduler.record_snapshot(f"Request {request.id} rejected")
            # Record failure?
            # reward = self.rl_optimizer.REWARD_FAILURE
            # self.rl_optimizer.remember(state, -1, reward, state, True, []) # Use dummy action -1?
            return False # Indicate failure

        # --- 3. RL 选择规则 (如果启用) ---
        chosen_rule_index = -1 # Default or placeholder
        if self.use_rl:
            chosen_rule_index = self.rl_optimizer.select_action(state, available_rule_indices)
            if chosen_rule_index is None:
                 print(f"[Warning Req {request.id}] RL failed to select a valid action from {available_rule_indices}. Falling back to first available rule.")
                 chosen_rule_index = available_rule_indices[0] # Fallback
            self.rl_decision_count += 1
        else:
            # 如果不使用 RL，默认使用第一个可用规则 (或者特定启发式规则)
            chosen_rule_index = available_rule_indices[0] # Example: default to first rule

        # --- 4. 执行选定的规则以找到车辆 ---
        selected_vehicle, selected_arrival_time = self._execute_rule(chosen_rule_index, request, current_time)

        # --- 5. 分配与经验记录 ---
        reward = self.rl_optimizer.REWARD_FAILURE # Default reward
        next_state_approx = state # Placeholder
        assignment_successful = False
        actual_arrival_time = None
        is_optimal_window = False
        vehicle_state_before = None 

        if selected_vehicle and selected_arrival_time is not None:
             # Store vehicle previous location *before* potential assignment
             # This is crucial for reward calculation involving travel time
             selected_vehicle.previous_location = selected_vehicle.current_location # Store current location as previous

             # Attempt assignment using the base scheduler's logic
             assignment_successful, actual_arrival_time = self.base_scheduler.assign_request(selected_vehicle, request)

             if assignment_successful and actual_arrival_time is not None:
                 request.is_served = True
                 is_optimal_window = (request.ei <= actual_arrival_time <= request.li)
                 
                 # 记录车辆到达和完成服务事件
                 self.base_scheduler.record_snapshot(f"V{selected_vehicle.id} arrive→R{request.id}")
                 self.base_scheduler.record_snapshot(f"V{selected_vehicle.id} finish→R{request.id}")
                 
                 # Pass necessary parameters to calculate_reward
                 reward = self.rl_optimizer.calculate_reward(
                     request,
                     selected_vehicle,
                     actual_arrival_time,
                     assignment_successful=True,
                     is_optimal=is_optimal_window,
                     base_scheduler_c1=self.base_scheduler.c1, # Pass c1
                     base_scheduler_c2=self.base_scheduler.c2, # Pass c2
                     current_time=current_time                 # Pass current_time
                 )

                 if request in self.base_scheduler.rejected_requests:
                     self.base_scheduler.rejected_requests.remove(request)
                 if request not in self.base_scheduler.served_requests:
                     self.base_scheduler.served_requests.append(request)

                 # Get Next State (Approximation) - After assignment
                 next_vehicles_state = self.base_scheduler.vehicles
                 next_current_time = actual_arrival_time # Or next request's ready_time? Use current for simplicity now
                 next_state_approx = self.rl_optimizer.encode_state(next_vehicles_state, request, next_current_time) # Encode with updated state

                 # Determine Available Next Actions (Approximation)
                 # This is tricky for VRP. For now, assume all rules are potentially available for the *next* request.
                 next_available_rule_indices_approx = list(range(self.rl_optimizer.NUM_RULES))

                 # Store Experience
                 if self.use_rl and self.mode == "train":
                #  if self.use_rl == True:
                     self.rl_optimizer.remember(
                         state, chosen_rule_index, reward, next_state_approx,
                         is_last_request, next_available_rule_indices_approx
                     )
                     # Experience Replay
                     self.rl_optimizer.replay()
             else: # Assignment failed by base_scheduler.assign_request
                 print(f"[Warning Req {request.id}] assign_request failed for V{selected_vehicle.id} chosen by rule {chosen_rule_index}.")
                 reward = self.rl_optimizer.REWARD_FAILURE
                 request.is_served = False
                 if request in self.base_scheduler.served_requests: self.base_scheduler.served_requests.remove(request)
                 if request not in self.base_scheduler.rejected_requests: self.base_scheduler.rejected_requests.append(request)
                 # 记录请求被拒绝事件
                 self.base_scheduler.record_snapshot(f"Request {request.id} rejected")
                 # Store failure experience
                 if self.use_rl:
                      next_state_fail = state # State didn't change
                      self.rl_optimizer.remember(state, chosen_rule_index, reward, next_state_fail, True, []) # Done=True, No next actions

        else: # Rule execution failed to find a vehicle
             print(f"[Info Req {request.id}] Rule {chosen_rule_index} failed to find a suitable vehicle. Rejecting.")
             self.base_scheduler.rejected_requests.append(request)
             request.is_served = False
             # 记录请求被拒绝事件
             self.base_scheduler.record_snapshot(f"Request {request.id} rejected")
             reward = self.rl_optimizer.REWARD_FAILURE
             # Store failure experience
             if self.use_rl:
                 next_state_fail = state # State didn't change
                 self.rl_optimizer.remember(state, chosen_rule_index, reward, next_state_fail, True, []) # Done=True, No next actions
                 self.rl_optimizer.log_reward(reward, self.rl_optimizer.train_counter)
                 self.log_epsilon(self.epsilon, self.train_counter)


        # Update target network periodically (already done inside replay)
        # if self.use_rl:
        #      self.rl_optimizer.update_target_model() # update_target_model is called within replay

        return assignment_successful # Return True if assignment succeeded

    def process_requests(self, requests):
        """处理一组充电请求 (RL 增强版入口 - 基于规则) """
        self.base_scheduler.reset()
        self.base_scheduler.all_requests = copy.deepcopy(requests)
        start_time = time.time()
        sorted_requests = sorted(self.base_scheduler.all_requests, key=lambda r: r.request_time)
        
        num_requests = len(sorted_requests)
        processed_count = 0
        failed_count = 0
        for i, request in enumerate(sorted_requests):
            self.base_scheduler.update_system_time(request.request_time)
            is_last = (i == num_requests - 1)
            success = self.process_request(request, is_last_request=is_last)
            if success:
                processed_count += 1
            else:
                failed_count += 1

            for vehicle in self.base_scheduler.vehicles:
                # 车辆类里应该有 remaining_travel_distance 和 remaining_service_capacity
                low_range = vehicle.remaining_travel_distance < vehicle.low_range_threshold
                low_cap   = vehicle.remaining_service_capacity < vehicle.low_capacity_threshold
                if low_range or low_cap:
                    # 先让车回充电站
                    vehicle.return_to_station(completion_time=vehicle.current_time)
                    # 再模拟充电
                    vehicle.recharge()            
            
        end_time = time.time()
        print(f"RL Enhanced Processing: Processed {processed_count}, Failed/Rejected {failed_count}")
        
        # 处理结束后让所有车辆返回充电站
        final_completion_time = self.base_scheduler.current_time
        for vehicle in self.base_scheduler.vehicles:
             if not vehicle.is_idle:
                 final_completion_time = max(final_completion_time, vehicle.available_time)
        
        for vehicle in self.base_scheduler.vehicles:
             vehicle.update_time(final_completion_time)
             vehicle.return_to_station(completion_time=final_completion_time)
        
        # 返回结果
        result = {
            'served_requests': self.base_scheduler.served_requests,
            'rejected_requests': self.base_scheduler.rejected_requests,
            'processing_time': end_time - start_time,
        }
        
        return result
    
    def get_statistics(self):
        """获取调度结果统计信息"""
        stats = self.base_scheduler.get_statistics()
        return stats
    
    def print_statistics(self, file=sys.stdout):
        """打印统计结果"""
        self.base_scheduler.print_statistics(file=file)

    def print_vehicle_routes(self, file=sys.stdout):
        """打印每个充电车的访问路径"""
        self.base_scheduler.print_vehicle_routes(file=file)
    
    def print_request_service_summary(self, file=sys.stdout):
        """打印每个请求的服务情况汇总"""
        self.base_scheduler.print_request_service_summary(file=file)
    
    def save_rl_model(self, filepath):
        """保存RL模型"""
        self.rl_optimizer.save_model(filepath)
    
    def load_rl_model(self, filepath):
        """加载RL模型"""
        self.rl_optimizer.load_model(filepath) 