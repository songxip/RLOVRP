import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import copy
from scheduler import OnlineScheduler
from rl_optimizer import RLEnhancedScheduler

class PerformanceEvaluator:
    """性能评估器类"""
    
    def __init__(self, data_generator=None):
        self.data_generator = data_generator
        self.results = {}
        self.raw_results = {}
    
    def record_result(self, dataset_name, algo_type, stats, scheduler, vehicles, requests, result_details):
        """记录一次实验运行的结果"""
        if dataset_name not in self.results:
            self.results[dataset_name] = {}
            self.raw_results[dataset_name] = {}
            
        self.results[dataset_name][algo_type] = stats
        # Save raw results for potential deeper analysis or visualization later
        self.raw_results[dataset_name][algo_type] = {
            'scheduler': scheduler,
            'vehicles': vehicles,
            'requests': requests,
            'result_details': result_details
        }
        
    def compare_algorithms(self, dataset_name):
        """比较指定数据集上两种算法的性能"""
        if dataset_name not in self.results or 'base' not in self.results[dataset_name] or 'rl' not in self.results[dataset_name]:
            print(f"错误：数据集 {dataset_name} 的基础或RL结果不完整，无法比较。")
            return None
        
        base_stats = self.results[dataset_name]['base']
        rl_stats = self.results[dataset_name]['rl']
        
        comparison = {
            'base': base_stats,
            'rl': rl_stats,
            'improvements': {}
        }
        
        # 计算改进百分比
        for metric in ['service_rate', 'optimal_rate', 'total_cost']:
            base_val = base_stats.get(metric, 0)
            rl_val = rl_stats.get(metric, 0)
            
            if metric == 'total_cost':
                # Lower is better for cost
                if base_val != 0:
                    improvement = (base_val - rl_val) / base_val
                elif rl_val != 0:
                    improvement = -float('inf') # Base was 0, RL is not (worse)
                else:
                    improvement = 0.0 # Both are 0
            else:
                # Higher is better for rates
                if base_val != 0:
                    improvement = (rl_val - base_val) / base_val
                elif rl_val != 0:
                    improvement = float('inf') # Base was 0, RL is not (better)
                else:
                    improvement = 0.0 # Both are 0
                
            comparison['improvements'][metric] = improvement
            
        return comparison
    
    def compare_all_datasets(self):
        """在所有数据集上比较算法性能"""
        datasets = list(self.results.keys())
        comparisons = {}
        
        for dataset in datasets:
            comparisons[dataset] = self.compare_algorithms(dataset)
        
        return comparisons
    
    def calculate_competitive_ratio(self, dataset_name):
        """计算竞争比（在线算法与理想离线算法的比值）"""
        # 对于移动充电车问题，离线最优解很难确定
        # 这里使用一个简化的方法，假设理想情况下所有请求都能被服务且在最优时间窗内
        if f"{dataset_name}_base" not in self.results:
            self.record_result(dataset_name, 'base', {}, None, None, None, None)
        
        stats = self.results[f"{dataset_name}_base"]['stats']
        
        # 假设离线最优解的时间成本只包含必要的移动时间和充电时间，没有惩罚时间
        # 并且所有请求都能被服务
        requests = self.results[f"{dataset_name}_base"]['requests']
        total_requests = len(requests)
        
        # 计算竞争比
        service_ratio = stats['served_requests'] / total_requests
        optimal_window_ratio = stats['optimal_window_served'] / stats['served_requests'] if stats['served_requests'] > 0 else 0
        
        # 时间成本竞争比（假设理想情况无惩罚时间）
        ideal_time_cost = stats['total_travel_time'] + stats['total_charging_time']
        time_cost_ratio = stats['total_cost'] / ideal_time_cost if ideal_time_cost > 0 else float('inf')
        
        return {
            'service_ratio': service_ratio,
            'optimal_window_ratio': optimal_window_ratio,
            'time_cost_ratio': time_cost_ratio
        }
    
    def plot_comparison(self, metric, title, ylabel, save_path=None):
        """绘制不同数据集和算法间的性能对比图"""
        datasets = sorted(list(self.results.keys()))
        base_values = []
        rl_values = []
        valid_datasets = []
        
        for dataset in datasets:
            if 'base' in self.results[dataset] and 'rl' in self.results[dataset]:
                base_val = self.results[dataset]['base'].get(metric)
                rl_val = self.results[dataset]['rl'].get(metric)
                if base_val is not None and rl_val is not None:
                    base_values.append(base_val)
                    rl_values.append(rl_val)
                    valid_datasets.append(dataset)
                else:
                    print(f"警告: 跳过数据集 {dataset} 的指标 '{metric}' 绘图，因为数据缺失。")
            
        if not valid_datasets:
            print(f"没有足够的数据用于绘制指标 '{metric}' 的比较图。")
            return

        # 创建柱状图
        x = np.arange(len(valid_datasets))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, base_values, width, label='基础启发式算法')
        rects2 = ax.bar(x + width/2, rl_values, width, label='RL增强算法')
        
        # 添加标签和标题
        ax.set_xlabel('数据集规模')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets)
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects, ax):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(rect.get_x() + rect.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(rects1, ax)
        autolabel(rects2, ax)
        
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存到: {save_path}")
            except Exception as e:
                print(f"保存图表失败: {e}")
        else:
            plt.show()
        plt.close(fig)
    
    def generate_report(self, save_path=None):
        """生成包含所有数据集比较结果的综合报告 (CSV 和 图表) - 使用 self.results """
        report_data = []
        
        for dataset in sorted(list(self.results.keys())):
            if 'base' in self.results[dataset] and 'rl' in self.results[dataset]:
                comp = self.compare_algorithms(dataset)
                if comp:
                    report_data.append({
                        'Dataset': dataset,
                        # Rates
                        'Base Service Rate (%)': comp['base'].get('service_rate', 0) * 100,
                        'RL Service Rate (%)': comp['rl'].get('service_rate', 0) * 100,
                        'Service Rate Improvement (%)': comp['improvements'].get('service_rate', 0) * 100,
                        'Base Optimal Rate (%)': comp['base'].get('optimal_rate', 0) * 100,
                        'RL Optimal Rate (%)': comp['rl'].get('optimal_rate', 0) * 100,
                        'Optimal Rate Improvement (%)': comp['improvements'].get('optimal_rate', 0) * 100,
                        # Costs
                        'Base Total Time Cost': comp['base'].get('total_cost', 0),
                        'RL Total Time Cost': comp['rl'].get('total_cost', 0),
                        'Time Cost Improvement (%)': comp['improvements'].get('total_cost', 0) * 100,
                        # --- Add Cost Components --- 
                        'Base Total Travel Time': comp['base'].get('total_travel_time', 0),
                        'RL Total Travel Time': comp['rl'].get('total_travel_time', 0),
                        'Base Total Charging Time': comp['base'].get('total_charging_time', 0),
                        'RL Total Charging Time': comp['rl'].get('total_charging_time', 0),
                        'Base Total Penalty Time': comp['base'].get('total_penalty_time', 0),
                        'RL Total Penalty Time': comp['rl'].get('total_penalty_time', 0),
                        # --------------------------
                        # Processing Time
                        'Base Processing Time (s)': comp['base'].get('processing_time', 0),
                        'RL Processing Time (s)': comp['rl'].get('processing_time', 0)
                    })
            
        if not report_data:
            print("没有足够的比较结果来生成报告。")
            return

        report_df = pd.DataFrame(report_data)
        report_df = report_df.set_index('Dataset')
        
        # --- Translate Headers for Console Output --- 
        console_headers_map = {
            'Base Service Rate (%)': '基础服务率(%)',
            'RL Service Rate (%)': 'RL服务率(%)',
            'Service Rate Improvement (%)': '服务率改进(%)',
            'Base Optimal Rate (%)': '基础最优率(%)',
            'RL Optimal Rate (%)': 'RL最优率(%)',
            'Optimal Rate Improvement (%)': '最优率改进(%)',
            'Base Total Time Cost': '基础总成本(h)',
            'RL Total Time Cost': 'RL总成本(h)',
            'Time Cost Improvement (%)': '成本改进(%)',
            'Base Total Travel Time': '基础行驶(h)',
            'RL Total Travel Time': 'RL行驶(h)',
            'Base Total Charging Time': '基础充电(h)',
            'RL Total Charging Time': 'RL充电(h)',
            'Base Total Penalty Time': '基础惩罚(h)',
            'RL Total Penalty Time': 'RL惩罚(h)',
            'Base Processing Time (s)': '基础耗时(s)',
            'RL Processing Time (s)': 'RL耗时(s)'
        }
        report_df_chinese = report_df.rename(columns=console_headers_map)
        # ---------------------------------------------
        
        print("\n--- 综合性能报告 --- ")
        # Print the DataFrame with Chinese headers to console
        print(report_df_chinese.to_string()) 
        
        # --- Save CSV and Plots (using original English headers for consistency in files) ---
        if save_path:
            csv_path = save_path + "_report.csv"
            plot_paths = {
                'service_rate': save_path + "_service_rate_comparison.png",
                'optimal_rate': save_path + "_optimal_rate_comparison.png",
                'total_cost': save_path + "_total_cost_comparison.png",
            }
            
            # 保存CSV (使用英文表头)
            try:
                report_df.to_csv(csv_path)
                print(f"报告已保存到 {csv_path}")
            except Exception as e:
                print(f"保存CSV报告失败: {e}")
        
            # 生成图表 (使用英文表头的数据和映射)
            try:
                metrics_to_plot = {
                    'service_rate': ('服务率对比 (%)', 'Service Rate (%)'),
                    'optimal_rate': ('最优时间窗服务率对比 (%)', 'Optimal Rate (%)'),
                    'total_cost': ('总时间成本对比', 'Total Cost')
                }
                
                for metric_key, (title, ylabel) in metrics_to_plot.items():
                    # Map internal metric key to DataFrame column names (original English)
                    if metric_key == 'service_rate':
                        base_col, rl_col = 'Base Service Rate (%)', 'RL Service Rate (%)'
                    elif metric_key == 'optimal_rate':
                        base_col, rl_col = 'Base Optimal Rate (%)', 'RL Optimal Rate (%)'
                    elif metric_key == 'total_cost':
                         base_col, rl_col = 'Base Total Time Cost', 'RL Total Time Cost'
                    else:
                        continue 
                    
                    # Check if columns exist before plotting
                    if base_col not in report_df.columns or rl_col not in report_df.columns:
                         print(f"警告: 跳过指标 '{metric_key}' 的绘图，因为报告DataFrame中缺少列。")
                         continue
                         
                    # Pass the original DataFrame (report_df) for plotting
                    self.plot_metric_comparison(report_df, base_col, rl_col, title, ylabel, plot_paths[metric_key])

            except Exception as e:
                import traceback
                print(f"生成图表失败: {e}")
                traceback.print_exc() 
                
    def plot_metric_comparison(self, report_df, base_col, rl_col, title, ylabel, save_path):
            """Helper function to plot a single metric comparison."""
            fig, ax = plt.subplots(figsize=(10, 6))
            index = np.arange(len(report_df.index))
            bar_width = 0.35

            rects1 = ax.bar(index - bar_width/2, report_df[base_col].values, bar_width, label='Base')
            rects2 = ax.bar(index + bar_width/2, report_df[rl_col].values, bar_width, label='RL')

            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(index)
            ax.set_xticklabels(report_df.index)
            ax.legend()
            ax.tick_params(axis='x', rotation=0) # No rotation needed if few datasets

            # Add value labels helper
            def autolabel(rects, ax):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
            autolabel(rects1, ax)
            autolabel(rects2, ax)
            
            plt.tight_layout()
            try:
                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
                 print(f"比较图表已保存到 {save_path}")
            except Exception as e:
                 print(f"保存图表 {save_path} 失败: {e}")
            plt.close(fig) # Close figure 