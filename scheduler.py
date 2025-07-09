import numpy as np
from utils import *
import copy
import time
import csv
import sys # Import sys for sys.stdout

class OnlineScheduler:
    """
    充电车在线调度算法
    启发式算法
    """
    
    def __init__(self, vehicles, c1=0.7, c2=0.8):
        self.vehicles = vehicles  # 所有充电车列表
        self.c1 = c1  # 早到惩罚系数
        self.c2 = c2  # 迟到惩罚系数
        self.served_requests = []  # 已服务的请求
        self.rejected_requests = []  # 被拒绝的请求
        self.current_time = 0.0  # 当前系统时间
        self.all_requests = None # 存储所有请求的列表
        self.timeline = []
    
    def reset(self):
        """重置调度器状态"""
        for vehicle in self.vehicles:
            vehicle.reset()
        self.served_requests.clear()
        self.rejected_requests.clear()
        self.timeline.clear()
        self.current_time = 0.0
        self.all_requests = None # 重置时也清除

    def reset_state(self):
        self.reset()
    
    def get_idle_vehicles(self):
        """获取当前空闲的充电车"""
        return [v for v in self.vehicles if v.is_idle]
    
    def get_busy_vehicles(self):
        """获取当前忙碌的充电车"""
        return [v for v in self.vehicles if not v.is_idle]
    
    def record_snapshot(self, label):
        """记录当前所有车辆状态到 timeline"""
        # 仿真时间：取所有车辆 current_time 的最大值
        t = max(v.current_time for v in self.vehicles)
        snap = {
            "time": t,
            "event": label,
            "vehicles": [
                {
                    "id": v.id,
                    "x": float(v.current_location[0]),
                    "y": float(v.current_location[1]),
                    "ratio": v.remaining_travel_distance / v.max_travel_distance,
                    "state": "idle" if v.is_idle else "busy"
                }
                for v in self.vehicles
            ]
        }
        self.timeline.append(snap)

    import csv

    def save_timeline_to_csv(self,timeline, path):
        """将timeline内容存入一个csv中"""
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            # 写表头
            writer.writerow(["time", "event", "vehicle_id", "x", "y", "ratio", "state"])
            # 写内容
            for snap in timeline:
                t = snap["time"]
                ev = snap["event"]
                for v in snap["vehicles"]:
                    writer.writerow([
                        t,
                        ev,
                        v["id"],
                        v["x"],
                        v["y"],
                        v["ratio"],
                        v["state"]
                    ])


    
    def process_request(self, request):
        """处理单个充电请求"""
        # 更新当前系统时间到请求发出时刻
        self.update_system_time(request.request_time)
        
        # 记录：收到新请求
        self.record_snapshot(f"Request {request.id} arrives")
      
        # 首先检查是否有空闲的充电车
        idle_vehicles = self.get_idle_vehicles()
        best_idle_vehicle = None
        best_idle_arrival_time = float('inf') # 记录最佳空闲车到达时间
        if idle_vehicles:
            # 选择最合适的空闲车辆 (不仅仅是最近，要考虑能否服务)
            best_idle_vehicle = self.select_best_idle_vehicle(idle_vehicles, request)
            if best_idle_vehicle:
                best_idle_arrival_time = best_idle_vehicle.estimate_arrival_time(request.location, self.current_time)

        # 考虑忙碌车辆
        busy_vehicles = self.get_busy_vehicles()
        best_busy_vehicle = None
        best_busy_arrival_time = float('inf')
        if busy_vehicles:
            candidate_busy_info = self.select_candidate_busy_vehicles(busy_vehicles, request)
            if candidate_busy_info:
                best_busy_vehicle, best_busy_arrival_time = self.select_best_busy_vehicle(candidate_busy_info, request)

        # 比较最佳空闲车和最佳忙碌车
        selected_vehicle = None
        # selected_arrival_time = float('inf') # 不再需要在这里记录，assign前确认

        if best_idle_vehicle and best_busy_vehicle:
            # 修改比较逻辑：选择预计到达时间更早的车辆
            # 注意：select_best_busy_vehicle 返回的 arrival_time 是根据规则选择后的最优值
            if best_idle_arrival_time <= best_busy_arrival_time:
                selected_vehicle = best_idle_vehicle
                # selected_arrival_time = best_idle_arrival_time
            else:
                selected_vehicle = best_busy_vehicle
                # selected_arrival_time = best_busy_arrival_time
        elif best_idle_vehicle:
            selected_vehicle = best_idle_vehicle
            # selected_arrival_time = best_idle_arrival_time
        elif best_busy_vehicle:
            selected_vehicle = best_busy_vehicle
            # selected_arrival_time = best_busy_arrival_time

        # 分配请求或拒绝
        if selected_vehicle:
            # 在最终分配前，做一次更严格的确认检查
            # 使用 vehicle.can_serve 进行最终确认
            if selected_vehicle.can_serve(request, self.current_time):
            # 打点：即将出发
                # self.record_snapshot(f"V{selected_vehicle.id} depart→R{request.id}")
                assigned, _ = self.assign_request(selected_vehicle, request)
                if assigned:
                    self.record_snapshot(f"V{selected_vehicle.id} arrive→R{request.id}")
                    finish_time = selected_vehicle.current_time
                    self.record_snapshot(f"V{selected_vehicle.id} finish→R{request.id}")
                    return True # 分配成功
                # else: 分配内部失败， fall-through 到拒绝逻辑
                #     # print(f"警告: 车辆 {selected_vehicle.id} 分配请求 {request.id} 失败 (assign_request 返回 False)")
                #     pass
            # else: # 最终检查失败，fall-through 到拒绝逻辑
            #      # print(f"信息: 车辆 {selected_vehicle.id} 在最终检查时无法服务请求 {request.id} ({format_time(selected_vehicle.estimate_arrival_time(request.location, self.current_time))} vs [{format_time(request.sei)}-{format_time(request.sli)}])")
            #      pass

        # 如果没有选到车，或者最终检查/分配失败，则拒绝
        self.rejected_requests.append(request)
        request.is_served = False
        self.record_snapshot(f"Request {request.id} rejected")
        return False
    
    def update_system_time(self, request_time):
        """更新系统时间到请求发出时刻，并更新车辆状态"""
        if request_time < self.current_time:
            # 允许处理稍早的请求，但不回退时间
            pass
            # print(f"警告: 请求时间({format_time(request_time)})早于当前系统时间({format_time(self.current_time)}). 检查数据或逻辑。")

        # 更新系统时间为 max(当前时间, 请求时间)
        effective_time = max(self.current_time, request_time)

        # 更新每辆车的状态（主要是检查是否变为空闲）
        for vehicle in self.vehicles:
            vehicle.update_time(effective_time) # 使用有效时间更新

        self.current_time = effective_time # 更新调度器的当前时间
    
    def select_best_idle_vehicle(self, idle_vehicles, request):
        """选择能够服务请求的最佳空闲车辆 (基于某种策略，例如最快到达)"""
        best_vehicle = None
        min_arrival_time = float('inf')

        for vehicle in idle_vehicles:
             # 使用 can_serve 检查基本约束和时间窗可行性
             # 传递当前调度器时间
             if vehicle.can_serve(request, self.current_time):
                 # 估计到达时间
                 arrival_time = vehicle.estimate_arrival_time(request.location, self.current_time)
                 # 选择最早到达的空闲车辆
                 if arrival_time < min_arrival_time:
                     min_arrival_time = arrival_time
                     best_vehicle = vehicle

        return best_vehicle
    
    def select_candidate_busy_vehicles(self, busy_vehicles, request):
        """筛选出能够服务新请求的忙碌车辆及其预计到达时间"""
        candidate_vehicles_info = []

        for vehicle in busy_vehicles:
            # 检查车辆是否能服务这个请求 (考虑当前任务完成时间和约束)
            # 传递当前调度器时间
            if vehicle.can_serve(request, self.current_time):
                # 估计到达时间
                arrival_time = vehicle.estimate_arrival_time(request.location, self.current_time)
                # 确保估计时间仍在可接受范围内 (can_serve 已检查 sli, 这里再次确认)
                if arrival_time <= request.sli: # 再次确认上界
                    candidate_vehicles_info.append((vehicle, arrival_time))
                # else:
                    # print(f"调试: 忙碌车辆 {vehicle.id} can_serve通过但estimate_arrival_time {format_time(arrival_time)} > sli {format_time(request.sli)}")

        return candidate_vehicles_info
    
    def select_best_busy_vehicle(self, candidate_vehicles_info, request):
        """根据时间窗情况选择最合适的忙碌车辆 (返回车辆和预计到达时间)"""
        optimal_window_vehicles = []
        early_window_vehicles = []
        late_window_vehicles = []

        for vehicle, arrival_time in candidate_vehicles_info:
            # 再次确保 arrival_time 在 [sei, sli] 内
            if request.sei <= arrival_time <= request.sli:
                if request.ei <= arrival_time <= request.li:
                    optimal_window_vehicles.append((vehicle, arrival_time))
                elif request.sei <= arrival_time < request.ei:
                    early_window_vehicles.append((vehicle, arrival_time))
                elif request.li < arrival_time <= request.sli:
                    late_window_vehicles.append((vehicle, arrival_time))
            # else: # 到达时间不在 [sei, sli] 内，candidate 筛选时理论上已过滤
            #     print(f"警告: 车辆 {vehicle.id} 的估计到达时间 {format_time(arrival_time)} 不在请求 {request.id} 的可接受时间窗 [{format_time(request.sei)}, {format_time(request.sli)}] 内，进入了 select_best_busy_vehicle")


        # 按优先级选择
        if optimal_window_vehicles:
            # 最优时间窗内，选最早到达
            return min(optimal_window_vehicles, key=lambda pair: pair[1])
        elif early_window_vehicles:
            # 提前时间窗内，选最晚到达 (最接近 ei)
            return max(early_window_vehicles, key=lambda pair: pair[1])
        elif late_window_vehicles:
            # 延迟时间窗内，选最早到达
            return min(late_window_vehicles, key=lambda pair: pair[1])
        else:
            # 没有找到任何符合条件的忙碌车辆
            return None, float('inf')
    
    def assign_request(self, vehicle, request):
        """将请求分配给选定的车辆，调用 vehicle.serve_request"""
        # 传递当前系统时间给 serve_request
        # serve_request 返回 (True/False, arrival_time) 或 (False, None)
        assigned, arrival_time = vehicle.serve_request(request, self.current_time, self.c1, self.c2)

        if assigned:
            self.served_requests.append(request)
            # 请求的服务信息已在 vehicle.serve_request -> request.assign_service 中更新
            return True, arrival_time
        else:
            # 分配失败，请求保持未服务状态
            return False, None
    
    def process_requests(self, requests):
        """处理一组充电请求（按时间顺序）"""
        self.reset()
        self.all_requests = copy.deepcopy(requests) # 存储原始请求列表的深拷贝
        
        #记录当前时间戳（用于记录算法耗时）
        start_time = time.time()
        # 确保按请求时间排序
        sorted_requests = sorted(self.all_requests, key=lambda r: r.request_time)

        latest_event_time = 0.0 # 记录系统中发生的最晚事件时间

        for request in sorted_requests:
            # 更新最新事件时间（至少是请求发出时间）
            latest_event_time = max(latest_event_time, request.request_time)
            self.process_request(request) # 这会更新 self.current_time
            # 2) 每次服务完一个请求后，检查所有车辆
            for vehicle in self.vehicles:
                # 车辆类里应该有 remaining_travel_distance 和 remaining_service_capacity
                low_range = vehicle.remaining_travel_distance < vehicle.low_range_threshold
                low_cap   = vehicle.remaining_service_capacity < vehicle.low_capacity_threshold
                if low_range or low_cap:
                    # 先让车回充电站
                    vehicle.return_to_station(completion_time=vehicle.current_time)
                    # 再模拟充电
                    vehicle.recharge()

        end_time = time.time()
        
        # 使用 process_request 循环中更新的 self.current_time 作为事件结束的基础时间
        latest_event_time = max(latest_event_time, self.current_time)


        # 处理结束后，更新所有车辆状态到最后一个事件发生的时间点
        # 并计算最终的返回充电站时间
        final_completion_time = latest_event_time # 初始为最晚事件时间
        for vehicle in self.vehicles:
             vehicle.update_time(latest_event_time) # 先更新到最晚事件时间
             if not vehicle.is_idle:
                 # 如果车辆仍在忙碌（例如正在前往最后一个任务点或正在充电）
                 # 更新 final_completion_time 为车辆的最终可用时间
                 final_completion_time = max(final_completion_time, vehicle.available_time)

        # 所有任务完成后，让车辆基于 final_completion_time 返回充电站
        for vehicle in self.vehicles:
             # 再次确保车辆状态更新到最终完成时间点
             vehicle.update_time(final_completion_time)
             vehicle.return_to_station(completion_time=final_completion_time)

        return {
            'served_requests': self.served_requests,
            'rejected_requests': self.rejected_requests,
            'processing_time': end_time - start_time
        }
    
    def get_statistics(self):
        """计算调度结果的统计信息"""
        num_requests = len(self.all_requests) if self.all_requests else 0
        num_served = len(self.served_requests)
        num_rejected = len(self.rejected_requests)
        num_optimal = sum(1 for req in self.served_requests if req.in_optimal_window)

        total_travel_time = sum(v.total_travel_time for v in self.vehicles)
        total_charging_time = sum(v.total_charging_time for v in self.vehicles)
        total_penalty_time = sum(v.total_penalty_time for v in self.vehicles)
        total_cost = total_travel_time + total_charging_time + total_penalty_time

        service_rate = num_served / num_requests if num_requests > 0 else 0
        optimal_rate = num_optimal / num_served if num_served > 0 else 0
        
        # Note: Competitive Ratio calculation is complex and depends on 
        # knowing the optimal offline solution, which is not computed here.
        competitive_ratio = None 
        
        return {
            "num_requests": num_requests,
            "num_served": num_served,
            "num_rejected": num_rejected,
            "num_optimal": num_optimal,
            "service_rate": service_rate,
            "optimal_rate": optimal_rate,
            "total_travel_time": total_travel_time,
            "total_charging_time": total_charging_time,
            "total_penalty_time": total_penalty_time,
            "total_cost": total_cost,
            "competitive_ratio": competitive_ratio
        }
    
    def print_vehicle_routes(self, file=sys.stdout):
        """打印每个充电车的访问路径 (使用 history) - 支持输出到文件"""
        print("\n--- 车辆路径详情 ---", file=file)
        if not self.vehicles:
            print("无车辆信息可打印路径。", file=file)
            return
        
        for vehicle in self.vehicles:
            print(f"\n充电车 {vehicle.id} (初始位置: {vehicle.start_location}):", file=file)
            
            # Ensure history exists and starts correctly
            if not vehicle.history or vehicle.history[0]['type'] != 'station' or vehicle.history[0]['arrival_time'] != 0:
                print(f"  -> 初始状态 @ 充电站 {vehicle.start_location}, 时间: 0.00", file=file) # Format time
                if vehicle.history:
                    print("  (原始历史记录可能不完整或格式错误)", file=file)
                else:
                    print("  (未执行任何任务)", file=file)
                    continue # Skip rest if no history

            # Print history events
            last_departure_time = 0.0
            for i, visit in enumerate(vehicle.history):
                loc_str = f"[{visit['location'][0]:.1f}, {visit['location'][1]:.1f}]"
                arr_str = f"{visit['arrival_time']:.2f}"
                dep_str = f"{visit['departure_time']:.2f}" if visit['departure_time'] is not None else "(结束)"

                # Check time continuity
                if visit['arrival_time'] < last_departure_time - 1e-6: # Allow for small float error
                    print(f"    时间警告: T{i} 到达 ({arr_str}) < 上次离开 ({last_departure_time:.2f}) ", file=file)

                event_desc = ""
                if visit['type'] == 'station':
                     # Should be caught by 'return' type now, or initial state
                     if i == 0 and visit['arrival_time'] == 0 and visit['departure_time'] == 0:
                          event_desc = f"T{i}: 初始 @ 充电站"
                     else:
                          # This case might indicate an issue if it appears mid-route
                          event_desc = f"T{i}: 异常充电站事件?" 
                elif visit['type'] == 'request':
                    event_desc = f"T{i}: 服务请求 {visit['request_id']}"
                elif visit['type'] == 'return':
                     event_desc = f"T{i}: 返回充电站"
                     # dep_str is already set to "(结束)"
                else:
                    event_desc = f"T{i}: 未知 ({visit['type']})"

                print(f"  {event_desc:<25} | 位置: {loc_str:<15} | 到达: {arr_str:<8} | 离开: {dep_str:<8}", file=file)

                if visit['departure_time'] is not None:
                     if visit['departure_time'] < visit['arrival_time'] - 1e-6:
                          print(f"    时间警告: T{i} 离开 ({dep_str}) < 到达 ({arr_str})", file=file)
                     last_departure_time = visit['departure_time']
                elif visit['type'] == 'return': # If it's the final return
                    last_departure_time = visit['arrival_time'] # End time is arrival at station
            print("--------------------", file=file)

    def print_request_service_summary(self, file=sys.stdout):
        """打印每个请求的服务情况汇总 - 支持输出到文件"""
        print("\n--- 请求服务情况汇总 ---", file=file)
        if self.all_requests is None:
            print("无请求数据可供汇总。", file=file)
            return

        # 确保按ID排序以便查找
        sorted_requests = sorted(self.all_requests, key=lambda r: r.id)
        
        for req in sorted_requests:
            status = "成功" if req.is_served else "失败"
            vehicle_id = req.assigned_vehicle_id if req.is_served else "N/A"
            optimal_status = "是" if req.is_served and req.in_optimal_window else "否"
            arrival_time_str = f"{req.actual_arrival_time:.2f}" if req.is_served else "N/A"
            start_time_str = f"{req.actual_start_time:.2f}" if req.is_served else "N/A"
            end_time_str = f"{req.actual_end_time:.2f}" if req.is_served else "N/A"
            penalty_str = f"{req.penalty_time:.2f} 小时" if req.is_served else "N/A"
            
            print(f"请求 ID: {req.id}", file=file)
            print(f"  发出时间: {req.request_time:.2f}", file=file)
            print(f"  最优窗: [{req.ei:.2f} - {req.li:.2f}]", file=file)
            print(f"  可接受窗: [{req.sei:.2f} - {req.sli:.2f}]", file=file)
            print(f"  服务结果: {status}", file=file)
            if req.is_served:
                print(f"  服务车辆 ID: {vehicle_id}", file=file)
                print(f"  实际到达时间: {arrival_time_str}", file=file)
                print(f"  实际服务开始: {start_time_str}", file=file)
                print(f"  实际服务结束: {end_time_str}", file=file)
                print(f"  是否在最优窗内: {optimal_status}", file=file)
                print(f"  惩罚时间成本: {penalty_str}", file=file)
            else:
                 pass
            print("-" * 10, file=file)
        print("-------------------------", file=file)
    
    def print_statistics(self, file=sys.stdout):
        """打印调度结果的统计信息 - 支持输出到文件"""
        stats = self.get_statistics()
        print("--- 调度结果统计 ---", file=file)
        print(f"总请求数: {stats['num_requests']}", file=file)
        print(f"成功服务数: {stats['num_served']}", file=file)
        print(f"拒绝服务数: {stats['num_rejected']}", file=file)
        print(f"服务率: {stats['service_rate']*100:.2f}%", file=file)
        print(f"最优时间窗内服务数: {stats['num_optimal']}", file=file)
        print(f"最优时间窗服务率 (占已服务): {stats['optimal_rate']*100:.2f}%", file=file)
        print("\n总时间成本明细:", file=file)
        print(f"  总行驶时间 (Tij): {stats['total_travel_time']:.2f} 小时", file=file)
        print(f"  总充电时间 (Tc): {stats['total_charging_time']:.2f} 小时", file=file)
        print(f"  总惩罚时间 (Tf): {stats['total_penalty_time']:.2f} 小时", file=file)
        print(f"  总成本 (Tij+Tc+Tf): {stats['total_cost']:.2f} 小时", file=file)
        print("--------------------", file=file) 