import numpy as np
import math
import matplotlib.pyplot as plt

# 计算两点之间的欧氏距离
def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

# 计算移动所需时间（小时）
def calculate_travel_time(point1, point2, speed=40):
    """
    计算两点间的移动时间
    point1, point2: 坐标点 (x, y)
    speed: 速度，单位为km/h
    返回时间，单位为小时
    """
    distance = calculate_distance(point1, point2)
    # 假设地图单位为km
    if speed <= 0:
        return float('inf')
    return distance / speed

# 计算充电所需时间（小时）
def calculate_charging_time(charge_amount, power=40):
    """
    计算充电所需时间
    charge_amount: 充电量，单位为kWh
    power: 充电功率，单位为kW
    返回时间，单位为小时
    """
    if power <= 0:
        return float('inf')

    current = 1 - charge_amount / 100
    target = 1

    # 计算积分公式
    numerator = 1 - 0.5 * current
    denominator = 1 - 0.5 * target

    v = float(140/100)
    # 执行积分计算
    time = (2.0 / v) * math.log(numerator / denominator)

    return time

# 计算惩罚时间
def calculate_penalty_time(arrival_time, ei, li, sei, sli, c1=0.7, c2=0.8):
    """
    计算时间窗惩罚
    arrival_time: 到达时间
    [ei, li]: 最优时间窗
    [sei, sli]: 可接受时间窗
    c1, c2: 惩罚系数
    """
    penalty = 0.0
    if sei <= arrival_time < ei:
        penalty = c1 * (ei - arrival_time)
    elif li < arrival_time <= sli:
        penalty = c2 * (arrival_time - li)
    # Outside [sei, sli] is handled by rejection, not penalty here
    return penalty

# 检查是否在最优时间窗内
def is_in_optimal_window(time, ei, li):
    return ei <= time <= li

# 检查是否在可接受时间窗内
def is_in_acceptable_window(time, sei, sli):
    return sei <= time <= sli 