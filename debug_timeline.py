#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试timeline数据的脚本
"""

from data_generator import DataGenerator
from scheduler import OnlineScheduler
from rl_optimizer import RLEnhancedScheduler
import copy

def debug_timeline_data():
    """调试timeline数据生成"""
    print("=== 调试Timeline数据生成 ===")
    
    # 加载数据
    data_loader = DataGenerator()
    vehicles, requests, map_size = data_loader.load_solomon_instance("C101_25.txt", 2)
    
    print(f"加载了 {len(requests)} 个请求和 {len(vehicles)} 辆车")
    
    # 测试基础调度器
    print("\n--- 测试基础调度器 ---")
    base_vehicles = copy.deepcopy(vehicles)
    base_requests = copy.deepcopy(requests)
    base_scheduler = OnlineScheduler(base_vehicles)
    base_result = base_scheduler.process_requests(base_requests)
    
    print(f"基础调度器 timeline 长度: {len(base_scheduler.timeline) if hasattr(base_scheduler, 'timeline') else 'N/A'}")
    if hasattr(base_scheduler, 'timeline') and base_scheduler.timeline:
        print("前5个timeline事件:")
        for i, event in enumerate(base_scheduler.timeline[:5]):
            print(f"  {i}: {event}")
    
    # 测试RL调度器
    print("\n--- 测试RL调度器 ---")
    rl_vehicles = copy.deepcopy(vehicles)
    rl_requests = copy.deepcopy(requests)
    rl_base_scheduler = OnlineScheduler(rl_vehicles)
    rl_scheduler = RLEnhancedScheduler(rl_base_scheduler, len(vehicles), use_rl=True, mode="single")
    rl_result = rl_scheduler.process_requests(rl_requests)
    
    print(f"RL调度器 timeline 长度: {len(rl_scheduler.base_scheduler.timeline) if hasattr(rl_scheduler.base_scheduler, 'timeline') else 'N/A'}")
    if hasattr(rl_scheduler.base_scheduler, 'timeline') and rl_scheduler.base_scheduler.timeline:
        print("前5个timeline事件:")
        for i, event in enumerate(rl_scheduler.base_scheduler.timeline[:5]):
            print(f"  {i}: {event}")
    
    # 检查车辆history
    print("\n--- 检查车辆History ---")
    for i, vehicle in enumerate(base_scheduler.vehicles):
        print(f"基础调度器 - 车辆 {i} history 长度: {len(vehicle.history) if hasattr(vehicle, 'history') else 'N/A'}")
        if hasattr(vehicle, 'history') and vehicle.history:
            print(f"  前3个历史事件: {vehicle.history[:3]}")
    
    for i, vehicle in enumerate(rl_scheduler.base_scheduler.vehicles):
        print(f"RL调度器 - 车辆 {i} history 长度: {len(vehicle.history) if hasattr(vehicle, 'history') else 'N/A'}")
        if hasattr(vehicle, 'history') and vehicle.history:
            print(f"  前3个历史事件: {vehicle.history[:3]}")

if __name__ == "__main__":
    debug_timeline_data() 