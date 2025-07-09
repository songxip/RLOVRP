# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from matplotlib.patches import Circle, Rectangle
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm

# # 设置支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# class ScheduleVisualizer:
#     """调度结果可视化器"""
    
#     def __init__(self, vehicles, requests, map_size):
#         self.vehicles = vehicles
#         self.requests = requests
#         self.map_size = map_size
#         self.colors = list(mcolors.TABLEAU_COLORS)
        
#     def plot_map(self, figsize=(10, 10)):
#         """绘制地图和充电站位置"""
#         plt.figure(figsize=figsize)
        
#         # 绘制地图边界
#         plt.xlim(0, self.map_size[0])
#         plt.ylim(0, self.map_size[1])
#         plt.grid(True, linestyle='--', alpha=0.7)
        
#         # 绘制充电站位置
#         station_location = self.vehicles[0].depot_location
#         plt.scatter(station_location[0], station_location[1], 
#                    color='red', s=200, marker='*', label='充电站')
        
#         # 添加标题和标签
#         plt.title('充电车调度地图')
#         plt.xlabel('X 坐标 (km)')
#         plt.ylabel('Y 坐标 (km)')
#         plt.legend()
        
#         return plt.gca()
    
#     def plot_requests(self, ax=None):
#         """绘制充电请求点"""
#         if ax is None:
#             ax = plt.gca()
        
#         # 绘制所有请求点
#         for request in self.requests:
#             if request.is_served:
#                 marker = 'o'  # 成功服务的请求
#                 color = 'green' if request.in_optimal_window else 'orange'
#                 label = '最优时间窗内服务' if request.in_optimal_window else '非最优时间窗内服务'
#             else:
#                 marker = 'x'  # 未服务的请求
#                 color = 'red'
#                 label = '未服务请求'
            
#             plt.scatter(request.location[0], request.location[1], 
#                        color=color, s=100, marker=marker, alpha=0.7, label=label)
        
#         # 去除重复的标签
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys())
        
#         return ax
    
#     def plot_vehicle_routes(self, ax=None):
#         """绘制车辆路径"""
#         if ax is None:
#             ax = plt.gca()
        
#         for i, vehicle in enumerate(self.vehicles):
#             color = self.colors[i % len(self.colors)]
            
#             # 绘制路径线
#             route = [vehicle.station_location] + vehicle.service_route
#             if len(route) > 1:
#                 x_coords = [loc[0] for loc in route]
#                 y_coords = [loc[1] for loc in route]
#                 ax.plot(x_coords, y_coords, color=color, linewidth=2, 
#                        label=f'车辆 {i} 路径', alpha=0.7)
                
#                 # 添加箭头指示方向
#                 for j in range(len(route)-1):
#                     dx = route[j+1][0] - route[j][0]
#                     dy = route[j+1][1] - route[j][1]
#                     if dx != 0 or dy != 0:  # 确保有移动
#                         plt.arrow(route[j][0], route[j][1], dx*0.8, dy*0.8, 
#                                  head_width=min(3, min(self.map_size)/50), 
#                                  head_length=min(5, min(self.map_size)/30), 
#                                  fc=color, ec=color, alpha=0.7)
        
#         # 去除重复的标签
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys())
        
#         return ax
    
#     def plot_schedule(self, figsize=(12, 8)):
#         """绘制调度甘特图"""
#         plt.figure(figsize=figsize)
        
#         # 创建颜色映射
#         cmap = cm.get_cmap('tab10', len(self.vehicles))
        
#         # 为每个车辆创建一行
#         y_positions = np.arange(len(self.vehicles))
        
#         # 设置图表参数
#         plt.grid(True, axis='x', linestyle='--', alpha=0.7)
#         plt.yticks(y_positions, [f'车辆 {v.id}' for v in self.vehicles])
#         plt.xlabel('时间 (小时)')
#         plt.ylabel('充电车')
#         plt.title('充电车调度甘特图')
        
#         # 绘制每辆车的调度
#         for i, vehicle in enumerate(self.vehicles):
#             # 绘制移动和服务时间段
#             for j in range(len(vehicle.visit_times)):
#                 start_time, end_time = vehicle.visit_times[j]
                
#                 if j < len(vehicle.requests_served):
#                     request_id = vehicle.requests_served[j]
#                     request = next((r for r in self.requests if r.id == request_id), None)
                    
#                     if request:
#                         # 绘制移动时间（从上一个位置到当前位置）
#                         prev_end = vehicle.visit_times[j-1][1] if j > 0 else 0
#                         if start_time > prev_end:
#                             # 移动时间段
#                             plt.barh(y_positions[i], start_time - prev_end, left=prev_end, 
#                                     height=0.5, color='gray', alpha=0.5, label='移动时间')
                        
#                         # 判断服务时间窗状态
#                         if request.in_optimal_window:
#                             color = 'green'  # 最优时间窗
#                             label = '最优时间窗服务'
#                         else:
#                             color = 'orange'  # 非最优时间窗
#                             label = '非最优时间窗服务'
                        
#                         # 绘制服务时间段
#                         plt.barh(y_positions[i], end_time - start_time, left=start_time, 
#                                 height=0.5, color=color, alpha=0.7, label=label)
                        
#                         # 添加请求ID标签
#                         plt.text(start_time + (end_time - start_time)/2, y_positions[i], 
#                                 f'请求 {request_id}', ha='center', va='center', color='black')
#                 else:
#                     # 最后一段是返回充电站
#                     prev_end = vehicle.visit_times[j-1][1] if j > 0 else 0
#                     plt.barh(y_positions[i], start_time - prev_end, left=prev_end, 
#                             height=0.5, color='gray', alpha=0.5, label='返回充电站')
        
#         # 去除重复的标签
#         handles, labels = plt.gca().get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         plt.legend(by_label.values(), by_label.keys(), loc='upper center', 
#                   bbox_to_anchor=(0.5, -0.15), ncol=3)
        
#         # 调整布局
#         plt.tight_layout()
#         plt.subplots_adjust(bottom=0.2)
        
#         return plt.gca()
    
#     def plot_time_window_distribution(self, figsize=(12, 6)):
#         """绘制时间窗服务分布"""
#         # 计算服务分类
#         optimal_served = sum(1 for r in self.requests if r.is_served and r.in_optimal_window)
#         non_optimal_served = sum(1 for r in self.requests if r.is_served and not r.in_optimal_window)
#         not_served = sum(1 for r in self.requests if not r.is_served)
        
#         # 创建饼图
#         plt.figure(figsize=figsize)
#         labels = ['最优时间窗内服务', '非最优时间窗内服务', '未服务']
#         sizes = [optimal_served, non_optimal_served, not_served]
#         colors = ['green', 'orange', 'red']
#         explode = (0.1, 0, 0)  # 突出最优时间窗内服务
        
#         plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
#                autopct='%1.1f%%', shadow=True, startangle=140)
#         plt.axis('equal')  # 保持饼图是圆形
#         plt.title('充电请求服务分布')
        
#         return plt.gca()
    
#     def plot_time_cost_breakdown(self, figsize=(10, 6)):
#         """绘制时间成本明细"""
#         # 计算各项时间成本
#         travel_time = sum(v.total_travel_time for v in self.vehicles)
#         charging_time = sum(v.total_charging_time for v in self.vehicles)
#         penalty_time = sum(v.total_penalty_time for v in self.vehicles)
        
#         # 创建柱状图
#         plt.figure(figsize=figsize)
#         categories = ['移动时间', '充电时间', '惩罚时间', '总时间成本']
#         values = [travel_time, charging_time, penalty_time, travel_time + charging_time + penalty_time]
        
#         bars = plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])
        
#         # 添加数值标签
#         for bar in bars:
#             height = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
#                     f'{height:.2f}h', ha='center', va='bottom')
        
#         plt.title('时间成本明细')
#         plt.ylabel('时间 (小时)')
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
        
#         return plt.gca()
    
#     def visualize_all(self, save_path=None):
#         """生成所有可视化图表"""
#         # 创建地图视图
#         plt.figure(figsize=(12, 10))
#         ax = self.plot_map()
#         self.plot_requests(ax)
#         self.plot_vehicle_routes(ax)
#         if save_path:
#             plt.savefig(f"{save_path}_map.png", dpi=300, bbox_inches='tight')
#         # plt.show()
        
#         # 创建甘特图
#         self.plot_schedule()
#         if save_path:
#             plt.savefig(f"{save_path}_schedule.png", dpi=300, bbox_inches='tight')
#         # plt.show()
        
#         # 创建时间窗分布饼图
#         self.plot_time_window_distribution()
#         if save_path:
#             plt.savefig(f"{save_path}_time_window.png", dpi=300, bbox_inches='tight')
#         # plt.show()
        
#         # 创建时间成本明细柱状图
#         self.plot_time_cost_breakdown()
#         if save_path:
#             plt.savefig(f"{save_path}_time_cost.png", dpi=300, bbox_inches='tight')
#         # plt.show()


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import font_manager
import numpy as np
import os
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from scheduler import *

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

class ScheduleVisualizer:
    """调度结果可视化器"""

    def __init__(self, vehicles, requests, map_size):
        self.vehicles = vehicles
        self.requests = requests
        self.map_size = map_size
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def plot_map(self, figsize=(10, 10)):
        """绘制地图和充电站位置"""
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # 绘制地图边界
        ax.set_xlim(0, self.map_size[0])
        ax.set_ylim(0, self.map_size[1])
        ax.grid(True, linestyle='--', alpha=0.7)

        # 绘制充电站位置（假设所有车辆同一站点）
        station = self.vehicles[0].depot_location
        ax.scatter(station[0], station[1],
                   color='red', s=200, marker='*', label='充电站')

        ax.set_title('充电车调度地图')
        ax.set_xlabel('X 坐标 (km)')
        ax.set_ylabel('Y 坐标 (km)')
        ax.legend()
        return ax

    def plot_requests(self, ax=None):
        """绘制充电请求点"""
        if ax is None:
            ax = plt.gca()

        used = set()
        for req in self.requests:
            if req.is_served and req.in_optimal_window:
                key, color, marker, label = 'opt', 'green', 'o', '最优时间窗内服务'
            elif req.is_served:
                key, color, marker, label = 'nonopt', 'orange', 'o', '非最优时间窗内服务'
            else:
                key, color, marker, label = 'unserved', 'red', 'x', '未服务请求'

            if key in used:
                lbl = None
            else:
                used.add(key)
                lbl = label

            ax.scatter(req.location[0], req.location[1],
                       c=color, s=100, marker=marker, alpha=0.7, label=lbl)

        # 清理重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        return ax

    def plot_vehicle_routes(self, ax=None):
        """绘制车辆路径（基于 history）"""
        if ax is None:
            ax = plt.gca()

        for i, v in enumerate(self.vehicles):
            color = self.colors[i % len(self.colors)]
            # 从 history 中提取轨迹
            xs = [ev['location'][0] for ev in v.history]
            ys = [ev['location'][1] for ev in v.history]
            ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.7, label=f'车辆 {v.id} 路径')

            # 添加箭头指示方向
            for j in range(len(xs) - 1):
                dx = xs[j+1] - xs[j]
                dy = ys[j+1] - ys[j]
                if dx != 0 or dy != 0:
                    ax.arrow(xs[j], ys[j], dx*0.8, dy*0.8,
                             head_width=min(3, min(self.map_size)/50),
                             head_length=min(5, min(self.map_size)/30),
                             fc=color, ec=color, alpha=0.7)

        # 清理重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        return ax

    def plot_schedule(self, figsize=(12, 8)):
        """绘制调度甘特图（需先在 vehicle 上生成 visit_times 和 requests_served）"""
        plt.figure(figsize=figsize)
        ax = plt.gca()
        cmap = cm.get_cmap('tab10', len(self.vehicles))
        y_positions = np.arange(len(self.vehicles))

        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'车辆 {v.id}' for v in self.vehicles])
        ax.set_xlabel('时间 (小时)')
        ax.set_ylabel('充电车')
        ax.set_title('充电车调度甘特图')

        for i, v in enumerate(self.vehicles):
            # 每段 visit_times[i] 是 (start, end)
            for j, (start, end) in enumerate(v.visit_times):
                # 移动或返回
                if j < len(v.requests_served):
                    req_id = v.requests_served[j]
                    r = next((r for r in self.requests if r.id == req_id), None)
                    # 移动段
                    prev_end = v.visit_times[j-1][1] if j > 0 else 0
                    if start > prev_end:
                        ax.barh(y_positions[i], start - prev_end, left=prev_end,
                                height=0.5, color='gray', alpha=0.5, label='移动时间')
                    # 服务段
                    color = 'green' if r and r.in_optimal_window else 'orange'
                    lbl = '最优时间窗服务' if r and r.in_optimal_window else '非最优时间窗服务'
                    ax.barh(y_positions[i], end - start, left=start,
                            height=0.5, color=color, alpha=0.7, label=lbl)
                    if r:
                        ax.text(start + (end - start)/2, y_positions[i],
                                f'{r.id}', ha='center', va='center', color='black')
                else:
                    # 最后一段：返回充电站
                    prev_end = v.visit_times[j-1][1] if j > 0 else 0
                    ax.barh(y_positions[i], start - prev_end, left=prev_end,
                            height=0.5, color='gray', alpha=0.5, label='返回充电站')

        # 清理重复图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center',
                  bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        return ax

    def plot_time_window_distribution(self, figsize=(12, 6)):
        """绘制时间窗服务分布"""
        optimal = sum(1 for r in self.requests if r.is_served and r.in_optimal_window)
        nonopt = sum(1 for r in self.requests if r.is_served and not r.in_optimal_window)
        unserved = sum(1 for r in self.requests if not r.is_served)

        plt.figure(figsize=figsize)
        labels = ['最优时间窗内服务', '非最优时间窗内服务', '未服务']
        sizes = [optimal, nonopt, unserved]
        colors = ['green', 'orange', 'red']
        explode = (0.1, 0, 0)

        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('充电请求服务分布')
        return plt.gca()

    def plot_time_cost_breakdown(self, figsize=(10, 6)):
        """绘制时间成本明细"""
        travel = sum(v.total_travel_time for v in self.vehicles)
        charge = sum(v.total_charging_time for v in self.vehicles)
        penalty = sum(v.total_penalty_time for v in self.vehicles)

        plt.figure(figsize=figsize)
        categories = ['移动时间', '充电时间', '惩罚时间', '总时间成本']
        values = [travel, charge, penalty, travel + charge + penalty]
        bars = plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])

        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                     f'{h:.2f}h', ha='center', va='bottom')

        plt.title('时间成本明细')
        plt.ylabel('时间 (小时)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        return plt.gca()

    def visualize_all(self, save_path=None, create_video=True, timeline_data=None):
        """生成所有可视化图表，包括动态视频"""
        # 地图路径图
        ax = self.plot_map()
        self.plot_requests(ax)
        self.plot_vehicle_routes(ax)
        if save_path:
            plt.savefig(f"{save_path}_map.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 甘特图
        ax = self.plot_schedule()
        if save_path:
            plt.savefig(f"{save_path}_schedule.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 时间窗分布
        ax = self.plot_time_window_distribution()
        if save_path:
            plt.savefig(f"{save_path}_time_window.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 时间成本明细
        ax = self.plot_time_cost_breakdown()
        if save_path:
            plt.savefig(f"{save_path}_time_cost.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 创建动态视频
        if create_video and save_path:
            print("正在生成动态视频...")
            self.create_animation_video(timeline_data=timeline_data, save_path=save_path)
            print("动态视频生成完成！")

    def create_animation_video(self, timeline_data=None, save_path=None, fps=5, duration_seconds=30):
        """
        创建动态视频可视化
        
        Args:
            timeline_data: 时间线数据，如果为None则从调度器的timeline获取
            save_path: 视频保存路径
            fps: 帧率
            duration_seconds: 视频总时长（秒）
        """
        if timeline_data is None:
            # 如果没有提供timeline数据，则基于车辆history生成简化的动画
            return self._create_simple_animation(save_path, fps, duration_seconds)
        
        # 设置图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制基础地图
        ax.set_xlim(-5, self.map_size[0] + 5)
        ax.set_ylim(-5, self.map_size[1] + 5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('充电车调度动态可视化', fontsize=16, fontweight='bold')
        ax.set_xlabel('X 坐标 (km)', fontsize=12)
        ax.set_ylabel('Y 坐标 (km)', fontsize=12)
        
        # 绘制充电站
        station = self.vehicles[0].depot_location
        ax.scatter(station[0], station[1], color='red', s=300, marker='*', 
                  label='充电站', zorder=10)
        
        # 绘制所有请求点（初始状态为灰色）
        request_scatters = {}
        for req in self.requests:
            scatter = ax.scatter(req.location[0], req.location[1], 
                               c='lightgray', s=150, marker='o', alpha=0.5, zorder=5)
            request_scatters[req.id] = scatter
        
        # 初始化车辆位置和轨迹
        vehicle_scatters = []
        vehicle_trails = []
        vehicle_trail_data = []
        
        for i, vehicle in enumerate(self.vehicles):
            color = self.colors[i % len(self.colors)]
            # 车辆当前位置
            scatter = ax.scatter(station[0], station[1], 
                               c=color, s=200, marker='s', 
                               label=f'车辆 {vehicle.id}', zorder=8)
            vehicle_scatters.append(scatter)
            
            # 车辆轨迹线
            line, = ax.plot([], [], color=color, linewidth=3, alpha=0.7, zorder=6)
            vehicle_trails.append(line)
            vehicle_trail_data.append([(station[0], station[1])])
        
        # 时间显示
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.8))
        
        # 状态显示
        status_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10,
                             verticalalignment='top', bbox=dict(boxstyle='round',
                             facecolor='lightblue', alpha=0.8))
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # 动画函数
        def animate(frame):
            if frame >= len(timeline_data):
                return []
            
            snapshot = timeline_data[frame]
            current_time = snapshot['time']
            event = snapshot['event']
            
            # 更新时间显示
            time_text.set_text(f'时间: {current_time:.2f}h')
            status_text.set_text(f'事件: {event}')
            
            # 更新车辆位置和轨迹
            for i, vehicle_data in enumerate(snapshot['vehicles']):
                if i < len(vehicle_scatters):
                    # 更新车辆位置
                    vehicle_scatters[i].set_offsets([[vehicle_data['x'], vehicle_data['y']]])
                    
                    # 更新轨迹
                    vehicle_trail_data[i].append((vehicle_data['x'], vehicle_data['y']))
                    trail_x = [pos[0] for pos in vehicle_trail_data[i]]
                    trail_y = [pos[1] for pos in vehicle_trail_data[i]]
                    vehicle_trails[i].set_data(trail_x, trail_y)
                    
                    # 根据车辆状态改变颜色透明度
                    alpha = 1.0 if vehicle_data['state'] == 'busy' else 0.6
                    vehicle_scatters[i].set_alpha(alpha)
            
            # 更新请求状态（如果事件涉及请求服务）
            if 'arrive' in event or 'finish' in event:
                # 解析请求ID
                try:
                    req_id = int(event.split('R')[-1])
                    if req_id in request_scatters:
                        if 'finish' in event:
                            # 服务完成，变为绿色
                            request_scatters[req_id].set_color('green')
                            request_scatters[req_id].set_alpha(1.0)
                        elif 'arrive' in event:
                            # 车辆到达，变为黄色
                            request_scatters[req_id].set_color('yellow')
                            request_scatters[req_id].set_alpha(1.0)
                except:
                    pass
            
            return []
        
        # 创建动画
        total_frames = min(len(timeline_data), fps * duration_seconds)
        if total_frames <= 0:
            total_frames = len(timeline_data)
        frame_interval = max(1, len(timeline_data) // total_frames) if total_frames > 0 else 1
        selected_frames = timeline_data[::frame_interval]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(selected_frames),
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存视频
        if save_path:
            try:
                # 保存为mp4格式
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='RLOVRP'), bitrate=1800)
                video_path = f"{save_path}_animation.mp4"
                anim.save(video_path, writer=writer)
                print(f"动画视频已保存至: {video_path}")
            except Exception as e:
                print(f"保存MP4失败: {e}")
                try:
                    # 尝试保存为gif格式
                    gif_path = f"{save_path}_animation.gif"
                    anim.save(gif_path, writer='pillow', fps=fps)
                    print(f"动画GIF已保存至: {gif_path}")
                except Exception as e2:
                    print(f"保存GIF也失败: {e2}")
        
        plt.close()
        return anim

    def _create_simple_animation(self, save_path=None, fps=5, duration_seconds=30):
        """
        基于车辆history创建简化动画
        """
        # 设置图形
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制基础地图
        ax.set_xlim(-5, self.map_size[0] + 5)
        ax.set_ylim(-5, self.map_size[1] + 5)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_title('充电车路径动态可视化', fontsize=16, fontweight='bold')
        ax.set_xlabel('X 坐标 (km)', fontsize=12)
        ax.set_ylabel('Y 坐标 (km)', fontsize=12)
        
        # 绘制充电站
        station = self.vehicles[0].depot_location
        ax.scatter(station[0], station[1], color='red', s=300, marker='*', 
                  label='充电站', zorder=10)
        
        # 绘制请求点
        self.plot_requests(ax)
        
        # 准备动画数据
        all_vehicle_paths = []
        max_steps = 0
        
        for vehicle in self.vehicles:
            if hasattr(vehicle, 'history') and vehicle.history:
                path = [(ev['location'][0], ev['location'][1]) for ev in vehicle.history]
                all_vehicle_paths.append(path)
                max_steps = max(max_steps, len(path))
            else:
                # 如果没有history，创建简单路径
                path = [tuple(station)]
                all_vehicle_paths.append(path)
        
        if max_steps == 0:
            print("没有找到车辆移动数据，无法创建动画")
            plt.close()
            return None
        
        # 初始化车辆显示
        vehicle_dots = []
        vehicle_trails = []
        
        for i, vehicle in enumerate(self.vehicles):
            color = self.colors[i % len(self.colors)]
            # 车辆位置点
            dot = ax.scatter(station[0], station[1], c=color, s=200, marker='s',
                           label=f'车辆 {vehicle.id}', zorder=8)
            vehicle_dots.append(dot)
            
            # 车辆轨迹线
            line, = ax.plot([], [], color=color, linewidth=3, alpha=0.7, zorder=6)
            vehicle_trails.append(line)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # 动画函数
        def animate(frame):
            for i, (dot, trail, path) in enumerate(zip(vehicle_dots, vehicle_trails, all_vehicle_paths)):
                if frame < len(path):
                    # 更新车辆位置
                    x, y = path[frame]
                    dot.set_offsets([[x, y]])
                    
                    # 更新轨迹
                    trail_x = [pos[0] for pos in path[:frame+1]]
                    trail_y = [pos[1] for pos in path[:frame+1]]
                    trail.set_data(trail_x, trail_y)
            
            return vehicle_dots + vehicle_trails
        
        # 创建动画
        total_frames = min(max_steps, fps * duration_seconds)
        anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                     interval=1000//fps, blit=False, repeat=True)
        
        # 保存视频
        if save_path:
            try:
                # 保存为mp4格式
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='RLOVRP'), bitrate=1800)
                video_path = f"{save_path}_animation.mp4"
                anim.save(video_path, writer=writer)
                print(f"简化动画视频已保存至: {video_path}")
            except Exception as e:
                print(f"保存MP4失败: {e}")
                try:
                    # 尝试保存为gif格式
                    gif_path = f"{save_path}_animation.gif"
                    anim.save(gif_path, writer='pillow', fps=fps)
                    print(f"简化动画GIF已保存至: {gif_path}")
                except Exception as e2:
                    print(f"保存GIF也失败: {e2}")
        
        plt.close()
        return anim

