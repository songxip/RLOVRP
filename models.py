import numpy as np
from utils import *

class ChargingRequest:
    """移动充电需求类"""
    def __init__(self, request_id, request_time, location, charge_amount, 
                 optimal_window_start, optimal_window_end, window_extension=5/60):
        self.id = request_id
        self.request_time = request_time  # 请求发出时刻
        self.location = location  # 请求位置 (x, y)
        self.charge_amount = charge_amount  # 充电需求量 kWh
        
        # 最优时间窗
        self.ei = optimal_window_start
        self.li = optimal_window_end
        
        # 可接受时间窗（最优时间窗两侧各扩展window_extension小时）
        self.sei = self.ei - window_extension
        self.sli = self.li + window_extension
        
        # 实际服务信息
        self.assigned_vehicle_id = None
        self.actual_arrival_time = None
        self.actual_start_time = None # 对于充电来说，到达即开始
        self.actual_end_time = None
        self.is_served = False
        self.in_optimal_window = False
        self.penalty_time = 0.0

    def get_charging_time(self, power=80):
        """计算该请求的充电时间"""
        return calculate_charging_time(self.charge_amount, power)
    
    def calculate_arrival_penalty(self, arrival_time, c1=0.7, c2=0.8):
        """计算到达惩罚时间"""
        return calculate_penalty_time(arrival_time, self.ei, self.li, self.sei, self.sli, c1, c2)
    
    def assign_service(self, vehicle_id, arrival_time, charging_time, penalty_time, c1=0.7, c2=0.8):
        """记录服务详情"""
        self.assigned_vehicle_id = vehicle_id
        self.actual_arrival_time = arrival_time
        self.actual_start_time = arrival_time # 假设到达即可开始充电
        self.actual_end_time = arrival_time + charging_time
        self.is_served = True
        self.penalty_time = penalty_time # 惩罚时间由车辆计算并传入

        # 判断是否在最优时间窗
        self.in_optimal_window = (self.ei <= arrival_time <= self.li)

        # print(f"请求 {self.id} 服务信息更新: 车辆={vehicle_id}, 到达={format_time(arrival_time)}, 结束={format_time(self.actual_end_time)}, 最优窗=[{format_time(self.ei)}-{format_time(self.li)}], in_optimal={self.in_optimal_window}, penalty={penalty_time}")
    
    def __str__(self):
        status = "成功" if self.is_served else "失败"
        window_status = "在最优时间窗内" if self.in_optimal_window else "不在最优时间窗内"
        return (
            f"请求ID: {self.id}, 发出时间: {self.request_time:.2f}, "
            f"位置: ({self.location[0]:.1f},{self.location[1]:.1f}), 充电量: {self.charge_amount:.1f}kWh, "
            f"最优时间窗: [{self.ei:.2f}-{self.li:.2f}], "
            f"可接受时间窗: [{self.sei:.2f}-{self.sli:.2f}], "
            f"服务状态: {status}, {window_status}"
        )


class ChargingVehicle:
    """表示移动充电车辆"""
    def __init__(self, vehicle_id, start_location=(0, 0), depot_location=None, capacity=270,
                 charging_power=80, speed=60,
                 max_travel_distance=600, max_service_capacity=500,
                 start_time=0):
        """
        初始化充电车辆.

        Args:
            vehicle_id: 车辆唯一标识符.
            start_location: 车辆起始位置 (通常是充电站).
            depot_location: 充电站/仓库的位置 (用于计算返回距离). 如果为None, 则使用start_location.
            capacity: 电池总容量 (kWh). (暂时未使用，先保留)
            charging_power: 充电功率 (kW).
            speed: 车辆平均行驶速度 (km/h) -> 提高到 60.
            max_travel_distance: 车辆最大续航里程 (km) -> 提高到 600.
            max_service_capacity: 车辆一次出发最多能提供的总电量 (kWh) -> 提高到 500.
            start_time: 车辆可用的起始时间.
        """
        self.id = vehicle_id
        self.current_location = np.array(start_location)
        self.previous_location = np.array(start_location) # 初始化 previous_location
        self.start_location = np.array(start_location) # 保存起始位置
        # 设置仓库位置，如果未提供则默认为起始位置
        self.depot_location = np.array(depot_location) if depot_location is not None else np.array(start_location)
        self.capacity = capacity # 电池总容量
        self.charging_power = charging_power
        self.speed = speed # km/h
        self.max_travel_distance = max_travel_distance
        self.max_service_capacity = max_service_capacity # 服务容量限制
        self.low_range_threshold = 100.0    # km，低于这个里程就要回站
        self.low_capacity_threshold = 50.0  # kWh，低于这个电量就要回站

        self.current_time = start_time # ADDED: Initialize current_time
        self.current_charge_provided = 0 # 当前行程已提供的电量
        self.current_distance_traveled = 0 # 当前行程已行驶的距离

        self.route = [(start_location, start_time, start_time)] # (地点, 到达时间, 离开时间)
        self.served_requests = [] # 服务的请求ID列表
        self.status = "idle" # idle, charging, driving

        self.debug_logs = [] # 用于记录调试信息

    def reset(self):
        """重置车辆状态"""
        self.current_location = self.start_location.copy()
        self.previous_location = self.start_location.copy() # 重置 previous_location
        self.current_time = 0.0
        self.is_idle = True
        self.remaining_travel_distance = self.max_travel_distance
        self.remaining_service_capacity = self.max_service_capacity
        self.service_route = [self.start_location.copy()] # 路径从起始位置开始
        # depot_location 通常在重置时不改变，保持初始化值
        # self.depot_location = self.depot_location # No change needed
        self.visit_times = [(0.0, 0.0)] # (到达时间, 离开时间)
        self.requests_served = []
        self.total_travel_time = 0.0 #行驶的时间
        self.total_charging_time = 0.0 # 充电时间
        self.total_penalty_time = 0.0 # 惩罚时间
        # 记录历史用于打印路径
        self.history = [{'type': 'station', 'location': self.start_location.copy(), 'arrival_time': 0.0, 'departure_time': 0.0}]
    


    def recharge(self, travel_charging_rate=60,service_charging_rate=50):
        """
        在充电站把续航和服务能力补满，更新时间。
        charging_rate: kW，默认为 self.charging_power
        """
        # 一定要在充电站位置调用
        assert np.array_equal(self.current_location, self.depot_location), "必须在充电站才能补电！"

        # 计算服务电池充电时间
        service_needed = self.max_service_capacity - self.remaining_service_capacity
        service_charging_time = service_needed/ service_charging_rate if service_needed > 0 else 0.0

        travel_needed = self.max_travel_distance - self.remaining_travel_distance
        travel_charging_time = travel_needed / travel_charging_rate if travel_needed > 0 else 0.0

        charging_time  = max(service_charging_time,travel_charging_time)

        # 更新时间、续航和服务能力
        self.current_time += charging_time
        self.remaining_service_capacity = self.max_service_capacity
        self.remaining_travel_distance = self.max_travel_distance

        # 记录历史
        self.history.append({
            'type': 'recharge',
            'location': self.depot_location.copy(),
            'arrival_time': self.current_time - charging_time,
            'departure_time': self.current_time
        })
        # 状态恢复空闲
        self.is_idle = True

    def update_time(self, current_system_time):
        """同步车辆时间到系统时间，并检查是否变为空闲"""
        # 如果车辆忙碌，且其预计完成时间早于或等于当前系统时间，则变为空闲
        # 车辆的 current_time 代表它完成当前任务链的时间点
        if not self.is_idle and self.current_time <= current_system_time:
            self.is_idle = True
            # 空闲后，它的位置是最后一个任务点，可用时间是 self.current_time
            # 注意：不能将 self.current_time 直接设为 current_system_time，
            # 否则会丢失完成任务的精确时间。

    def travel_to(self, target_location, departure_time):
        """在指定的出发时间后，移动到目标位置。返回 (行驶时间, 到达时间)"""
        # 在移动前记录当前位置
        loc_before_move = self.current_location.copy()

        distance = calculate_distance(self.current_location, target_location)
        travel_time = calculate_travel_time(self.current_location, target_location, self.speed)
        arrival_time = departure_time + travel_time

        # 检查续航是否足够 (这里只检查单程，can_serve应检查往返)
        if distance > self.remaining_travel_distance:
             # print(f"车辆 {self.id} 续航不足以到达 {target_location}. 需要 {distance}, 剩余 {self.remaining_travel_distance}")
             return None, None # 表示无法移动

        # 更新状态
        self.current_location = target_location.copy()
        self.previous_location = loc_before_move # 更新 previous_location
        self.current_time = arrival_time # 车辆的内部时间更新为到达时间
        self.remaining_travel_distance -= distance
        self.total_travel_time += travel_time

        return travel_time, arrival_time

    def serve_request(self, request, current_system_time, c1=0.7, c2=0.8):
        """尝试服务充电请求，在检查通过后才更新状态"""

        # 1. 计算最早出发时间 (不能早于系统当前时间，也不能早于车辆完成上个任务的时间)
        departure_time = max(current_system_time, self.current_time)

        # 2. 计算行驶时间和预计到达时间
        travel_time = calculate_travel_time(self.current_location, request.location, self.speed)
        arrival_time = departure_time + travel_time

        # 3. 检查时间窗约束
        if not (request.sei <= arrival_time <= request.sli):
            # print(f"车辆 {self.id} 无法在可接受时间窗 [{format_time(request.sei)}, {format_time(request.sli)}] 内到达请求 {request.id} (预计到达: {format_time(arrival_time)})")
            return False, None # 分配失败

        # 4. 检查电量和续航约束 (调用 can_serve 进行详细检查)
        # 注意：can_serve 需要知道预计的 arrival_time 来做判断
        if not self.can_serve(request, current_system_time): # 传递 current_system_time
            # print(f"车辆 {self.id} 因约束无法服务请求 {request.id}")
            return False, None # 分配失败

        # --- 所有检查通过，现在执行服务 ---

        # 5. 执行移动 (更新位置, 车辆时间, 剩余续航, 累计行驶时间)
        # 使用计算好的 departure_time
        actual_travel_time, actual_arrival_time = self.travel_to(request.location, departure_time)

        # 再次确认移动成功 (续航可能在 travel_to 内部再次检查)
        if actual_travel_time is None:
             # 这个理论上不应发生，因为 can_serve 已经检查过
             print(f"警告: 车辆 {self.id} 在 travel_to 中失败，即使 can_serve 通过。检查逻辑。")
             # 需要回滚状态吗？ travel_to 设计为检查失败时不改变状态
             return False, None

        # 6. 计算惩罚时间
        penalty_time = calculate_penalty_time(actual_arrival_time, request.ei, request.li, request.sei, request.sli, c1, c2)
        self.total_penalty_time += penalty_time

        # 7. 计算充电时间并更新车辆时间
        charging_time = calculate_charging_time(request.charge_amount, self.charging_power)
        service_end_time = actual_arrival_time + charging_time
        self.current_time = service_end_time # 车辆完成服务的时间
        self.total_charging_time += charging_time

        # 8. 更新剩余服务电量
        self.remaining_service_capacity -= request.charge_amount

        # 9. 更新请求的服务信息
        request.assign_service(
             vehicle_id = self.id,
             arrival_time = actual_arrival_time,
             charging_time = charging_time,
             penalty_time = penalty_time,
             c1=c1, c2=c2 # 传递 c1, c2 以便 request 内部正确判断 in_optimal_window
        )

        # 10. 更新服务记录和历史
        # self.service_route.append(request.location.copy()) # travel_to 已更新 current_location
        # self.visit_times.append((actual_arrival_time, service_end_time)) # history 包含这些信息
        self.requests_served.append(request.id)
        self.history.append({
            'type': 'request',
            'request_id': request.id,
            'location': self.current_location.copy(),
            'arrival_time': actual_arrival_time,
            'departure_time': service_end_time
        })


        # 11. 更新车辆状态
        self.is_idle = False

        # print(f"车辆 {self.id} 成功分配请求 {request.id}. 到达: {format_time(actual_arrival_time)}, 离开: {format_time(service_end_time)}")
        return True, actual_arrival_time # 返回成功状态和实际到达时间

    def can_serve(self, request, current_system_time):
        """检查是否能够服务请求 (考虑时间和约束)"""
        # print(f"[Debug V{self.id}] Checking can_serve for Request {request.id} at system time {format_time(current_system_time)} (Vehicle available at {format_time(self.current_time)})")
        # 1. 检查剩余电量是否足够 (服务这个请求本身)
        if request.charge_amount > self.remaining_service_capacity:
             print(f"[Debug V{self.id}] Fail Req {request.id}: Low Capacity ({self.remaining_service_capacity:.1f} < {request.charge_amount:.1f})")
             return False

        # 2. 检查续航里程是否足够 (从当前位置 -> 请求位置 -> 充电站)
        dist_to_req = calculate_distance(self.current_location, request.location)
        dist_req_to_station = calculate_distance(request.location, self.depot_location)
        required_distance = dist_to_req + dist_req_to_station
        if required_distance > self.remaining_travel_distance:
             print(f"[Debug V{self.id}] Fail Req {request.id}: Low Range ({self.remaining_travel_distance:.1f} < {required_distance:.1f})")
             return False

        # 3. 检查时间窗可行性
        # 计算最早出发时间
        departure_time = max(current_system_time, self.current_time)
        # 计算预计到达时间
        travel_time = calculate_travel_time(self.current_location, request.location, self.speed)
        arrival_time = departure_time + travel_time

        # 必须在最晚可接受时间之前到达
        if arrival_time > request.sli:
             print(f"[Debug V{self.id}] Fail Req {request.id}: Arrival ({arrival_time:.2f}) > SLI ({request.sli:.2f}) (Dep: {departure_time:.2f}, Travel: {travel_time:.2f}h)")
             return False
        
        # --- 新增：检查是否早于最早可接受时间 ---
        if arrival_time < request.sei:
             print(f"[Debug V{self.id}] Fail Req {request.id}: Arrival ({arrival_time:.2f}) < SEI ({request.sei:.2f}) (Dep: {departure_time:.2f}, Travel: {travel_time:.2f}h)")
             return False
        # ---------------------------------------
        
        # print(f"[Debug V{self.id}] Pass Req {request.id}: Arrival {format_time(arrival_time)} in [{format_time(request.sei)}, {format_time(request.sli)}]\")
        # 如果能到达，则认为可以服务
        return True

    def estimate_arrival_time(self, request_location, current_system_time):
        """基于当前系统时间，估计到达请求位置的时间"""
        departure_time = max(current_system_time, self.current_time)
        travel_time = calculate_travel_time(self.current_location, request_location, self.speed)
        return departure_time + travel_time

    @property
    def available_time(self):
        """返回车辆下一次可用的时间 (完成当前任务链的时间)"""
        return self.current_time

    def return_to_station(self, completion_time=None):
        """让车辆返回充电站"""
        
        # 确定出发时间：优先使用传入的 completion_time，否则使用车辆当前的 available_time
        # completion_time 代表调度器认为的系统结束时间点或车辆完成最后一个任务的时间点
        departure_time = completion_time if completion_time is not None else self.available_time

        # 如果车辆当前位置就是充电站，并且出发时间为0（初始状态），则无需操作
        if np.array_equal(self.current_location, self.start_location) and departure_time == 0:
            # 确保历史记录是正确的初始状态
            if not self.history or not (self.history[0]['type'] == 'station' and self.history[0]['arrival_time'] == 0):
                self.reset() # 如果历史不对，重置以确保初始状态正确
            return

        # 如果车辆已经在充电站，但 departure_time > 0，说明它完成了任务并在站内结束
        # 我们需要确保历史记录反映了这一点
        if np.array_equal(self.current_location, self.start_location):
            # print(f"车辆 {self.id} 在 {format_time(departure_time)} 完成任务于充电站。")
            # 确保历史记录的最后一条是到达充电站且时间正确
            last_event = self.history[-1] if self.history else None
            if not last_event or last_event['type'] != 'station' or last_event['arrival_time'] != departure_time:
                 # 如果最后一条历史不是在 completion_time 到达充电站，添加一个标记
                 # 这通常发生在最后一个任务恰好是充电站位置的情况
                 self.history.append({
                       'type': 'return', # 标记为返回事件
                       'request_id': None,
                       'location': self.start_location.copy(),
                       'arrival_time': departure_time, # 到达时间就是完成时间
                       'departure_time': None # 不再离开
                   })
            # 更新车辆状态，确保时间正确且空闲
            self.current_time = departure_time
            self.is_idle = True
            return

        # print(f"车辆 {self.id} 开始从 {self.current_location} 返回充电站 (出发时间: {format_time(departure_time)})")

        # 计算行驶时间和到达时间
        distance = calculate_distance(self.current_location, self.depot_location)
        travel_time = calculate_travel_time(self.current_location, self.depot_location, self.speed)
        arrival_time = departure_time + travel_time

        # 更新状态
        self.total_travel_time += travel_time
        # 注意：这里可能需要检查剩余续航是否足够返回，但假设规划时已考虑
        self.remaining_travel_distance -= distance
        if self.remaining_travel_distance < 0:
             print(f"警告: 车辆 {self.id} 返回充电站后剩余续航为负 ({self.remaining_travel_distance:.2f}).")


        self.current_location = self.depot_location.copy()
        self.current_time = arrival_time # 车辆时间更新为到达充电站的时间
        self.is_idle = True # 返回后变为空闲

        # 添加返回记录到历史
        self.history.append({
            'type': 'return', # 使用 'return' 类型标记
            'request_id': None,
            'location': self.depot_location.copy(),
            'arrival_time': arrival_time,
            'departure_time': None # 返回后不再离开
        })
        # print(f"车辆 {self.id} 已返回充电站. 到达时间: {format_time(arrival_time)}, 总行驶时间: {self.total_travel_time:.2f}")
    
    def __str__(self):
        status = "空闲" if self.is_idle else "忙碌"
        loc_str = f"({self.current_location[0]:.1f},{self.current_location[1]:.1f})"
        return (
            f"充电车ID: {self.id}, 状态: {status}, 当前位置: {loc_str}, "
            f"当前时间: {self.current_time:.2f}, "
            f"剩余续航里程: {self.remaining_travel_distance:.2f}km, "
            f"剩余服务电量: {self.remaining_service_capacity:.2f}kWh, "
            f"已服务请求数: {len(self.requests_served)}"
        ) 