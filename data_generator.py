import numpy as np
import random
import pandas as pd # Add pandas for easier parsing
import re # Add regex for parsing
from models import ChargingRequest, ChargingVehicle
from utils import calculate_travel_time

class DataGenerator:
    """数据加载器类 (从Solomon文件)"""
    
    def __init__(self, seed=None): # Seed less relevant now
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        # 保持车辆参数与之前一致
        self.vehicle_params = {
            'speed': 60.0, 
            'max_travel_distance': 600.0, 
            'max_service_capacity': 500.0,
            'charging_power': 80.0
        }

    def load_solomon_instance(self, filepath, num_vehicles):
        """
        从Solomon格式文件加载实例数据。

        Args:
            filepath (str): Solomon实例文件的路径。
            num_vehicles (int): 该实例应使用的充电车数量。

        Returns:
            tuple: (vehicles, requests, map_size)
                   vehicles: 充电车列表
                   requests: 充电需求列表
                   map_size: (maxX, maxY) 根据坐标推断的地图大小
        """
        vehicles = []
        requests = []
        depot_location = None
        coords = []

        time_conversion_factor = 60.0 # 将文件中的时间单位（假设是分钟）转换为小时
        window_extension_h = 20.0 / time_conversion_factor # 20分钟扩展

        with open(filepath, 'r') as f:
            lines = f.readlines()

        customer_section_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 查找客户数据起始行
            if 'CUST NO.' in line and 'XCOORD.' in line:
                customer_section_started = True
                continue
                
            if not customer_section_started:
                continue

            # 解析客户数据行 (使用空格分割)
            parts = re.split(r'\s+', line) # 使用正则表达式匹配一个或多个空格
            if len(parts) < 7: # 确保行数据足够
                continue
            
            try:
                cust_no = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3]) # 假设为kWh
                ready_time = float(parts[4]) # 分钟
                due_date = float(parts[5])   # 分钟
                # service_time = float(parts[6]) # 忽略服务时间
            except ValueError:
                print(f"警告: 跳过无法解析的行: {line}")
                continue
            
            coords.append((x, y))

            if cust_no == 0:
                depot_location = np.array([x, y])
            else:
                # 创建充电请求
                request = ChargingRequest(
                    request_id=cust_no,
                    # request_time 设为 ready_time (小时)
                    request_time=ready_time / time_conversion_factor, 
                    location=np.array([x, y]),
                    charge_amount=demand,
                    # 最优时间窗 = [ready_time, due_date] (小时)
                    optimal_window_start=ready_time / time_conversion_factor,
                    optimal_window_end=due_date / time_conversion_factor,
                    # 扩展时间窗 (小时)
                    window_extension=window_extension_h
                )
                requests.append(request)

        if depot_location is None:
            raise ValueError(f"错误: 文件 {filepath} 中未找到仓库点 (CUST NO. 0)")

        # 创建充电车
        for i in range(num_vehicles):
            vehicle = ChargingVehicle(
                vehicle_id=i,
                start_location=depot_location.copy(),
                depot_location=depot_location.copy(), # 传递仓库位置
                speed=self.vehicle_params['speed'],
                max_travel_distance=self.vehicle_params['max_travel_distance'],
                max_service_capacity=self.vehicle_params['max_service_capacity'],
                charging_power=self.vehicle_params['charging_power']
            )
            vehicles.append(vehicle)
            
        # 计算地图大小 (稍微留些边距)
        if coords:
             coords_array = np.array(coords)
             min_x, min_y = coords_array.min(axis=0)
             max_x, max_y = coords_array.max(axis=0)
             map_size = (max_x + 10, max_y + 10) # 加一点边距
        else: 
             map_size = (100, 100) # 默认值

        # 按请求时间（即Ready Time）排序 - 保持在线特性
        requests.sort(key=lambda r: r.request_time)
        
        print(f"从 {filepath} 加载了 {len(requests)} 个请求和 {len(vehicles)} 辆车。仓库点: {depot_location}, 地图范围: ~{map_size}")
        return vehicles, requests, map_size

    # --- 不再需要之前的生成方法 --- 
    # def generate_dataset(...):
    #     pass
    # def generate_small_dataset(...):
    #     pass
    # def generate_medium_dataset(...):
    #     pass
    # def generate_large_dataset(...):
    #     pass

# --- (可选) 保留一个简单的测试部分 --- 
if __name__ == "__main__":
    loader = DataGenerator()
    
    # 测试加载 (假设文件在同级目录)
    try:
        print("测试加载 C101_25.txt...")
        vehicles, requests, map_size = loader.load_solomon_instance('D:\items_2rd_year\OnlineVRP__\OnlineVRP__\C101_25.txt', num_vehicles=2)
        print(f"  加载成功: {len(vehicles)} vehicles, {len(requests)} requests, map_size={map_size}")
        if requests:
             print(f"  第一个请求: {requests[0]}")
             print(f"  最后一个请求: {requests[-1]}")
    except FileNotFoundError:
        print("错误: 未找到 C101_25.txt")
    except Exception as e:
        print(f"加载 C101_25.txt 时出错: {e}")
        
    try:
        print("\n测试加载 C101_50.txt...")
        vehicles, requests, map_size = loader.load_solomon_instance('D:\items_2rd_year\OnlineVRP__\OnlineVRP__\C101_50.txt', num_vehicles=4)
        print(f"  加载成功: {len(vehicles)} vehicles, {len(requests)} requests, map_size={map_size}")
    except FileNotFoundError:
        print("错误: 未找到 C101_50.txt")
    except Exception as e:
        print(f"加载 C101_50.txt 时出错: {e}")

    try:
        print("\n测试加载 C101_100.txt...")
        vehicles, requests, map_size = loader.load_solomon_instance('D:\items_2rd_year\OnlineVRP__\OnlineVRP__\C101_100.txt', num_vehicles=6)
        print(f"  加载成功: {len(vehicles)} vehicles, {len(requests)} requests, map_size={map_size}")
    except FileNotFoundError:
        print("错误: 未找到 C101_100.txt")
    except Exception as e:
        print(f"加载 C101_100.txt 时出错: {e}") 