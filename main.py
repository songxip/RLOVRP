import os
import argparse
import time
import matplotlib.pyplot as plt
import copy # Added for deep copying loaded data
import sys # Import sys for file writing
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import random # Added for random shufflingP
import pandas as pd # Added for final report generation


os.system('chcp 65001')

from data_generator import DataGenerator
from models import ChargingRequest, ChargingVehicle
from scheduler import OnlineScheduler
from rl_optimizer import RLEnhancedScheduler
from visualizer import ScheduleVisualizer
from utils import *
# Removed PerformanceEvaluator import as it's no longer used directly here
# from evaluator import PerformanceEvaluator

# --- 全局常量 ---
DEFAULT_SAVE_PATH = "results"
RL_MODEL_FILENAME = "rl_model.keras" # 默认模型文件名
# Map dataset names to filenames and vehicle counts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG = {
    "small": {"filename": os.path.join(SCRIPT_DIR,"C101_25.txt"), "num_vehicles": 2},
    "medium": {"filename": os.path.join(SCRIPT_DIR,"C101_50.txt"), "num_vehicles": 4},
    "large": {"filename": os.path.join(SCRIPT_DIR,"C101_100.txt"), "num_vehicles": 6}
}
FINAL_REPORT_FILENAME = "final_comparison_report.csv" # Final report CSV filename
# ---------------

def run_single_test(data_loader, dataset_name, use_rl=False, visualize=True, save_path=DEFAULT_SAVE_PATH, load_model=False):
    """运行单个测试 (使用加载的数据)"""
    print(f"运行 {dataset_name} 数据集测试 (从文件加载), {'使用RL优化' if use_rl else '不使用RL优化'}")
    
    # 加载数据集
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"未知的 数据集名称: {dataset_name}")
        
    try:
        vehicles, requests, map_size = data_loader.load_solomon_instance(
            config["filename"], config["num_vehicles"]
        )
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {config['filename']}")
        return None, None, None, None # Return None on failure
    except Exception as e:
        print(f"加载数据文件 {config['filename']} 时出错: {e}")
        return None, None, None, None # Return None on failure

    # 创建基础调度器
    # Important: Use deepcopy for vehicles if they might be modified by the scheduler
    # and we need the original state later (less critical in single run).
    base_scheduler = OnlineScheduler(copy.deepcopy(vehicles))
    
    # 模型文件路径
    rl_model_path = os.path.join(save_path, RL_MODEL_FILENAME)
    
    # 根据是否使用RL选择调度器
    if use_rl:
        # Pass the base_scheduler instance to RLEnhancedScheduler
        scheduler = RLEnhancedScheduler(base_scheduler, len(vehicles), use_rl=True,mode = "single")
        result_prefix = f"{dataset_name}_rl"
        # 尝试加载模型
        if load_model:
            scheduler.load_rl_model(rl_model_path)
    else:
        scheduler = base_scheduler
        result_prefix = f"{dataset_name}_base"
    
    # 运行调度算法
    print("开始处理请求...")
    start_time = time.time()
    # Ensure scheduler uses the correct (potentially copied) vehicle list
    # scheduler.base_scheduler.vehicles = base_scheduler.vehicles # RLEnhancedScheduler takes base_scheduler
    result = scheduler.process_requests(copy.deepcopy(requests))
    end_time = time.time()
    print(f"请求处理完成，耗时: {end_time - start_time:.2f}秒")
    
    # 保存RL模型（如果使用了RL）
    if use_rl:
        scheduler.save_rl_model(rl_model_path)
      
    # 打印统计信息
    scheduler.print_statistics()
    
    # 打印车辆路径
    # Access vehicles from the scheduler after processing
    scheduler.print_vehicle_routes() 

    # 打印请求服务情况
    scheduler.print_request_service_summary()
    
    # 可视化结果
    if visualize:
        print("生成可视化结果...")
        vis_save_path = os.path.join(save_path, result_prefix) if save_path else None
        if save_path and not os.path.exists(os.path.dirname(vis_save_path)):
             os.makedirs(os.path.dirname(vis_save_path), exist_ok=True) # Use exist_ok=True
        # Pass the final state of vehicles from the scheduler to the visualizer
        # Ensure correct vehicle and request lists are passed
        final_vehicles = scheduler.base_scheduler.vehicles if isinstance(scheduler, RLEnhancedScheduler) else scheduler.vehicles
        if use_rl == True : 
            final_requests = scheduler.base_scheduler.all_requests
            timeline_data = scheduler.base_scheduler.timeline if hasattr(scheduler.base_scheduler, 'timeline') else None
        else :
            final_requests = scheduler.all_requests # Assuming all_requests holds the final state including unserved
            timeline_data = scheduler.timeline if hasattr(scheduler, 'timeline') else None
        visualizer = ScheduleVisualizer(final_vehicles, final_requests, map_size)
        visualizer.visualize_all(save_path=vis_save_path, create_video=True, timeline_data=timeline_data) # Suffix will be added by visualizer
    
    # Return the scheduler instance which holds the final state
    if use_rl == False : 
        return scheduler, scheduler.vehicles, scheduler.all_requests, map_size
    else :
        return scheduler, scheduler.base_scheduler.vehicles, scheduler.base_scheduler.all_requests, map_size

def run_comparative_test(data_loader, save_path=DEFAULT_SAVE_PATH):
    """运行比较测试, 确定最佳结果并生成报告和可视化 (使用加载的数据)"""
    print("运行比较测试，对比启发式算法与RL增强算法性能，并输出最佳结果 (从文件加载)...")

    rl_model_path = os.path.join(save_path, RL_MODEL_FILENAME)
    datasets = list(DATASET_CONFIG.keys())
    all_results_data = [] # 用于存储每个数据集的结果供最后报告

    for dataset in datasets:
        print(f"测试 {dataset} 数据集:")
        config = DATASET_CONFIG[dataset]
        filename = config["filename"]
        num_vehicles = config["num_vehicles"]

        results_txt_path = None
        vis_save_dir = None
        if save_path:
             results_txt_path = os.path.join(save_path, f"{dataset}_best_results.txt") # Changed filename
             vis_save_dir = save_path # Visualizations will be saved in the main save_path
             print(f"  结果将保存到: {results_txt_path}")
             if not os.path.exists(save_path):
                 os.makedirs(save_path, exist_ok=True)
        # -------------------------------

        try:
            all_vehicles, all_requests, map_size = data_loader.load_solomon_instance(filename, num_vehicles)
        except FileNotFoundError:
            print(f"错误: 找不到数据文件 {filename}，跳过此数据集")
            continue
        except Exception as e:
            print(f"加载数据文件 {filename} 时出错: {e}，跳过此数据集")
            continue

        # --- 运行基础启发式算法 ---
        print("  运行基础启发式算法...")
        base_vehicles_copy = copy.deepcopy(all_vehicles)
        base_requests_copy = copy.deepcopy(all_requests)
        base_scheduler = OnlineScheduler(base_vehicles_copy)
        base_result = base_scheduler.process_requests(base_requests_copy)
        base_stats = base_scheduler.get_statistics()
        base_stats['processing_time'] = base_result.get('processing_time', 0)
        print(f"    基础算法完成，服务率: {base_stats.get('service_rate', 0)*100:.2f}%，最优率: {base_stats.get('optimal_rate', 0)*100:.2f}%，总成本: {base_stats.get('total_cost', 0):.4f}h，处理时间: {base_stats['processing_time']:.2f}s")

        # --- 运行RL增强算法 ---
        print("  运行RL增强算法...")
        rl_vehicles_copy = copy.deepcopy(all_vehicles)
        rl_requests_copy = copy.deepcopy(all_requests)
        # RLEnhancedScheduler 需要一个 OnlineScheduler 实例
        rl_base_scheduler_instance = OnlineScheduler(rl_vehicles_copy)
        max_vehicles_for_rl = max(cfg["num_vehicles"] for cfg in DATASET_CONFIG.values()) # Ensure consistent RL state size
        rl_scheduler = RLEnhancedScheduler(rl_base_scheduler_instance, max_vehicles_for_rl, use_rl=True,mode = "compare")
        # 尝试加载模型，如果失败则继续（可能从头开始训练/决策）
        rl_scheduler.load_rl_model(rl_model_path)

        rl_result = rl_scheduler.process_requests(rl_requests_copy)
        rl_stats = rl_scheduler.get_statistics()
        rl_processing_time = rl_result.get('processing_time', 0)
        rl_stats['processing_time'] = rl_processing_time
        print(f"    RL算法完成，服务率: {rl_stats.get('service_rate', 0)*100:.2f}%，最优率: {rl_stats.get('optimal_rate', 0)*100:.2f}%，总成本: {rl_stats.get('total_cost', 0):.4f}h，处理时间: {rl_stats['processing_time']:.2f}s")

        # --- 保存更新后的模型 ---
        rl_scheduler.save_rl_model(rl_model_path)
        # ----------------------------

        # --- 确定最佳结果 ---
        print("  确定最佳结果...")
        is_rl_better = False
        # 优先级: 1. 服务率 2. 最优率 3. 总成本
        if rl_stats.get('service_rate', -1) > base_stats.get('service_rate', -1):
            is_rl_better = True
        elif rl_stats.get('service_rate', -1) == base_stats.get('service_rate', -1):
            if rl_stats.get('optimal_rate', -1) > base_stats.get('optimal_rate', -1):
                is_rl_better = True
            elif rl_stats.get('optimal_rate', -1) == base_stats.get('optimal_rate', -1):
                # 总成本越低越好
                if rl_stats.get('total_cost', float('inf')) < base_stats.get('total_cost', float('inf')):
                    is_rl_better = True

        if is_rl_better:
            best_scheduler = rl_scheduler
            best_stats = rl_stats
            best_vehicles = rl_scheduler.base_scheduler.vehicles # Get vehicles from the inner scheduler
            best_requests = rl_scheduler.base_scheduler.all_requests # Corrected line
            best_result_source = "RL Enhanced"
            print("    RL 算法结果更优。")
        else:
            best_scheduler = base_scheduler
            best_stats = base_stats
            best_vehicles = base_scheduler.vehicles
            best_requests = base_scheduler.all_requests
            best_result_source = "Base Heuristic"
            print("    基础启发式算法结果更优或持平。")
        # -----------------------

        # --- 保存最佳结果到 TXT 文件 ---
        if results_txt_path:
            try:
                with open(results_txt_path, 'w', encoding='utf-8') as f:
                    print(f"数据集: {dataset} ({filename}) - 最佳结果 ({best_result_source})", file=f) # Updated title

                    print("===============================", file=f)
                    print("=== 最佳算法结果统计 ===", file=f)
                    print("===============================", file=f)
                    best_scheduler.print_statistics(file=f)

                    print("===============================", file=f)
                    print("=== 最佳算法车辆路径 ===", file=f)
                    print("===============================", file=f)
                    best_scheduler.print_vehicle_routes(file=f)

                    print("===============================", file=f)
                    print("=== 最佳算法请求服务摘要 ===", file=f)
                    print("===============================", file=f)
                    best_scheduler.print_request_service_summary(file=f)

                    # print(f"(此文件显示的是 {best_result_source} 算法的结果)", file=f) # Add source info

                print(f"  最佳结果详细信息已写入 {results_txt_path}")
            except Exception as e:
                print(f"错误: 写入结果文件 {results_txt_path} 失败: {e}")
        # --------------------------------

        # --- 生成最佳结果的可视化 ---
        if vis_save_dir:
            print("  生成最佳结果的可视化...")
            vis_save_path_prefix = os.path.join(vis_save_dir, f"{dataset}_best") # Use prefix
            try:
                # 获取timeline数据用于视频生成
                best_timeline = best_scheduler.base_scheduler.timeline if hasattr(best_scheduler, 'base_scheduler') and hasattr(best_scheduler.base_scheduler, 'timeline') else (best_scheduler.timeline if hasattr(best_scheduler, 'timeline') else None)
                visualizer = ScheduleVisualizer(best_vehicles, best_requests, map_size)
                visualizer.visualize_all(save_path=vis_save_path_prefix, create_video=True, timeline_data=best_timeline) # Visualizer adds suffixes like _routes.png
                print(f"  可视化结果已保存至 {vis_save_dir} (前缀: {os.path.basename(vis_save_path_prefix)})")
            except Exception as e:
                print("")
                # print(f"错误: 生成或保存可视化文件失败: {e}")
        # ----------------------------

        # --- 收集数据用于最终报告 ---
        all_results_data.append({
            "Dataset": dataset,
            "Base Service Rate (%)": base_stats.get('service_rate', 0) * 100,
            "RL Service Rate (%)": rl_stats.get('service_rate', 0) * 100,
            "Base Optimal Rate (%)": base_stats.get('optimal_rate', 0) * 100,
            "RL Optimal Rate (%)": rl_stats.get('optimal_rate', 0) * 100,
            "Base Total Cost (h)": base_stats.get('total_cost', 0),
            "RL Total Cost (h)": rl_stats.get('total_cost', 0),
            "Base Processing Time (s)": base_stats.get('processing_time', 0),
            "RL Processing Time (s)": rl_stats.get('processing_time', 0),
            "Best Result Source": best_result_source
        })
        # -----------------------------

    # --- 生成最终比较报告 ---
    print("--- 生成最终综合性能报告 ---")
    if not all_results_data:
        print("没有足够的数据生成报告。")
        return None

    df = pd.DataFrame(all_results_data)

    # 计算改进率
    # 服务率改进 = (RL - Base) / Base (如果 Base > 0)
    df['Service Rate Improvement (%)'] = df.apply(
        lambda row: ((row['RL Service Rate (%)'] - row['Base Service Rate (%)']) / row['Base Service Rate (%)'] * 100)
                    if row['Base Service Rate (%)'] > 0 and row['RL Service Rate (%)'] >= row['Base Service Rate (%)'] else 0, # Only show positive improvement
        axis=1
    )
    # 最优率改进 = (RL - Base) / Base (如果 Base > 0)
    df['Optimal Rate Improvement (%)'] = df.apply(
        lambda row: ((row['RL Optimal Rate (%)'] - row['Base Optimal Rate (%)']) / row['Base Optimal Rate (%)'] * 100)
                    if row['Base Optimal Rate (%)'] > 0 and row['RL Optimal Rate (%)'] >= row['Base Optimal Rate (%)'] else 0, # Only show positive improvement
        axis=1
    )
    # 成本改进 = (Base - RL) / Base (因为成本越低越好, 如果 Base > 0)
    df['Cost Improvement (%)'] = df.apply(
        lambda row: ((row['Base Total Cost (h)'] - row['RL Total Cost (h)']) / row['Base Total Cost (h)'] * 100)
                    if row['Base Total Cost (h)'] > 0 and row['RL Total Cost (h)'] <= row['Base Total Cost (h)'] else 0, # Only show positive improvement (cost reduction)
        axis=1
    )

    # 选择最终报告的列，并根据最佳结果调整显示值
    report_df = pd.DataFrame()
    # report_df['Dataset'] = df['Dataset'] # Keep original for reference
    # report_df['Best Service Rate (%)'] = df.apply(lambda row: row['RL Service Rate (%)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Service Rate (%)'], axis=1)
    # report_df['Best Optimal Rate (%)'] = df.apply(lambda row: row['RL Optimal Rate (%)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Optimal Rate (%)'], axis=1)
    # report_df['Best Total Cost (h)'] = df.apply(lambda row: row['RL Total Cost (h)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Total Cost (h)'], axis=1)
    # report_df['Service Rate Improvement (%)'] = df['Service Rate Improvement (%)'] # Improvement is based on RL vs Base comparison
    # report_df['Optimal Rate Improvement (%)'] = df['Optimal Rate Improvement (%)'] # Improvement is based on RL vs Base comparison
    # report_df['Cost Improvement (%)'] = df['Cost Improvement (%)'] # Improvement is based on RL vs Base comparison
    # report_df['Best Processing Time (s)'] = df.apply(lambda row: row['RL Processing Time (s)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Processing Time (s)'], axis=1)
    # report_df['Result Source'] = df['Best Result Source'] # Indicate where the 'Best' values came from

    # --- Define columns with Chinese headers ---
    report_df['数据集'] = df['Dataset']
    report_df['最佳服务率 (%)'] = df.apply(lambda row: row['RL Service Rate (%)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Service Rate (%)'], axis=1)
    report_df['最佳最优率 (%)'] = df.apply(lambda row: row['RL Optimal Rate (%)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Optimal Rate (%)'], axis=1)
    report_df['最佳总成本 (h)'] = df.apply(lambda row: row['RL Total Cost (h)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Total Cost (h)'], axis=1)
    report_df['服务率改进 (%)'] = df['Service Rate Improvement (%)']
    report_df['最优率改进 (%)'] = df['Optimal Rate Improvement (%)']
    report_df['成本改进 (%)'] = df['Cost Improvement (%)']
    report_df['最佳处理时间 (s)'] = df.apply(lambda row: row['RL Processing Time (s)'] if row['Best Result Source'] == 'RL Enhanced' else row['Base Processing Time (s)'], axis=1)
    report_df['结果来源'] = df['Best Result Source']
    # -------------------------------------------

    # 格式化浮点数列
    # Adjust float_cols to match the new Chinese headers
    float_cols = ['最佳服务率 (%)', '最佳最优率 (%)', '最佳总成本 (h)',
                  '服务率改进 (%)', '最优率改进 (%)', '成本改进 (%)',
                  '最佳处理时间 (s)']
    for col in float_cols:
        report_df[col] = report_df[col].round(4) # Adjust precision as needed

    # 打印到控制台
    print("最终综合性能报告:")
    # Use to_string for better console formatting
    print(report_df.to_string(index=False))

    # 保存到 CSV
    if save_path:
        final_report_path = os.path.join(save_path, FINAL_REPORT_FILENAME)
        try:
            report_df.to_csv(final_report_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for Excel compatibility
            print(f"最终报告已保存到: {final_report_path}")
        except Exception as e:
            print(f"错误: 保存最终报告 CSV 文件失败: {e}")

    return report_df # Return the final report dataframe
# -------------------------

def run_training_loop(data_loader, num_epochs, train_dataset="medium", save_path=DEFAULT_SAVE_PATH):
    """运行RL训练循环 (使用加载的数据, 轮流训练所有规模)"""
    print(f"开始 RL 训练，共 {num_epochs} 轮，轮流使用所有规模数据集")

    rl_model_path = os.path.join(save_path, RL_MODEL_FILENAME)

    # 使用最大车辆数初始化调度器 (确保状态维度足够)
    max_vehicles = max(config["num_vehicles"] for config in DATASET_CONFIG.values())

    # --- 初始化 RLEnhancedScheduler ---
    # 创建一个临时的 base_scheduler 实例用于初始化 RLEnhancedScheduler
    # 注意: 这个 base_scheduler 实例的 vehicles 列表会在每个数据集加载时被替换
    temp_base_scheduler = OnlineScheduler([]) # Needs vehicles list
    rl_scheduler = RLEnhancedScheduler(temp_base_scheduler, max_vehicles, use_rl=True)
    # ----------------------------------

    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"--- 开始训练轮次 {epoch}/{num_epochs} --- ")
        # 加载当前模型 (以便继续训练)
        # 在每个 epoch 开始时加载一次
        rl_scheduler.load_rl_model(rl_model_path) # Load model weights and potentially epsilon
        print(f"当前 Epsilon: {rl_scheduler.rl_optimizer.epsilon:.4f}") # Check loaded epsilon

        datasets_to_train = list(DATASET_CONFIG.keys())
        random.shuffle(datasets_to_train)

        epoch_total_requests = 0
        epoch_total_served = 0
        # epoch_total_reward = 0 # Track reward if needed

        for dataset_name in datasets_to_train:
             print(f"  -- 训练数据集: {dataset_name} --")
             config = DATASET_CONFIG[dataset_name]
             filename = config["filename"]
             num_vehicles_train = config["num_vehicles"]

             try:
                 vehicles, requests, _ = data_loader.load_solomon_instance(filename, num_vehicles_train)
             except FileNotFoundError:
                 print(f"错误: 找不到数据文件 {filename}，跳过此数据集训练")
                 continue
             except Exception as e:
                 print(f"加载数据文件 {filename} 时出错: {e}，跳过此数据集训练")
                 continue

             # --- 更新 RLEnhancedScheduler 内部的 base_scheduler 状态 ---
             # 必须确保 RLEnhancedScheduler 使用的是当前数据集的车辆
             # 同时传递 c1 和 c2 给新的 OnlineScheduler 实例
             # Make sure the vehicles list is updated IN the scheduler
             # The RLEnhancedScheduler holds its own OnlineScheduler instance
             rl_scheduler.base_scheduler.vehicles = copy.deepcopy(vehicles)
             rl_scheduler.base_scheduler.num_vehicles = len(vehicles)
             rl_scheduler.base_scheduler.c1 = 0.7 # 早到惩罚系数
             rl_scheduler.base_scheduler.c2 = 0.8 # 迟到惩罚系数
             rl_scheduler.base_scheduler.reset_state() # Reset any internal state of the base scheduler

             # Update all_requests in the RL scheduler for the new dataset
             rl_scheduler.all_requests = [] # Reset requests for the new dataset run

             # RLOptimizer 的 state_dim 是固定的 (基于 max_vehicles),
             # encode_state 会处理实际车辆数量并填充/截断
             # ------------------------------------------------------

             print(f"    处理 {len(requests)} 个请求并训练...")
             start_time = time.time()
             # RLEnhancedScheduler 的 process_requests 方法应该处理整个列表
             # 它内部会调用 process_request 逐个处理，并在其中进行 RL 训练 (remember/replay)
             result_summary = rl_scheduler.process_requests(copy.deepcopy(requests))
             end_time = time.time()
             print(f"    数据集 {dataset_name} 处理完成，耗时: {end_time - start_time:.2f}秒")

             # 累积统计 (从返回的 summary 获取)
             stats = rl_scheduler.get_statistics() # get_statistics 应该基于当前的 base_scheduler
             epoch_total_requests += stats.get('num_requests', 0)
             epoch_total_served += stats.get('num_served', 0)
             # 奖励信息可能需要从 RLEnhancedScheduler 或 RLOptimizer 获取更详细的记录
             # 简单的：可以累加 process_requests 返回的总奖励 (如果它返回的话)
             # epoch_total_reward += result_summary.get('total_reward', 0) # Assuming process_requests returns this

        # --- Epoch 结束 ---
        if epoch_total_requests > 0:
             epoch_avg_service_rate = (epoch_total_served / epoch_total_requests) * 100
             print(f"  轮次 {epoch} 平均服务率: {epoch_avg_service_rate:.2f}%")
             # 可以打印平均奖励等信息

        # Epsilon 衰减 (现在由 RLOptimizer.replay 内部处理)
        # print(f"  Epsilon decayed to: {rl_scheduler.rl_optimizer.epsilon:.4f}") # epsilon 在 replay 中衰减

        # 保存更新后的模型 (每个 Epoch 结束时保存)
        print(f"保存模型到 {rl_model_path}...")
        rl_scheduler.save_rl_model(rl_model_path) # Save model weights and potentially epsilon

    print(f"--- 训练完成 --- 共 {num_epochs} 轮")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行在线车辆调度模拟")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "train"], help="运行模式: single(单个测试), compare(比较并输出最佳结果), train(RL训练)")
    parser.add_argument("--dataset", type=str, default="small", choices=list(DATASET_CONFIG.keys()), help="数据集大小 (用于 single 模式, 对应Solomon文件: small=C101_25, medium=C101_50, large=C101_100)")
    parser.add_argument("--use_rl", action='store_true', help="在 single 模式下使用 RL 优化")
    parser.add_argument("--load_model", action='store_true', help="在 single/train 模式下尝试加载预训练 RL 模型")
    parser.add_argument("--no_visualize", action='store_true', help="在 single/compare 模式下禁用可视化")
    parser.add_argument("--save_path", type=str, default=DEFAULT_SAVE_PATH, help="保存结果、模型和报告的路径")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数 (仅在 train 模式下有效)") # 增加默认值
    # train_dataset argument removed as training now iterates through all datasets
    # parser.add_argument("--train_dataset", type=str, default="medium", choices=list(DATASET_CONFIG.keys()), help="用于训练的数据集 (不再使用，训练将遍历所有数据集)")

    args = parser.parse_args()

    # 创建保存目录
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True) # Use exist_ok=True

    # 创建数据加载器
    data_loader = DataGenerator()

    if args.mode == "compare":
        # 运行比较测试并生成报告/可视化
        run_comparative_test(data_loader, save_path=args.save_path)
    elif args.mode == "single":
        # 运行单个测试
        run_single_test(
            data_loader, args.dataset, use_rl=args.use_rl,
            visualize=not args.no_visualize, save_path=args.save_path,
            load_model=args.load_model
        )
    elif args.mode == "train":
        # 运行训练循环
        run_training_loop(data_loader, args.epochs,
                          save_path=args.save_path) # Removed train_dataset arg
    else:
        # 这部分理论上不会执行，因为 choices 限制了参数
        print(f"错误：未知的运行模式 '{args.mode}'")


if __name__ == "__main__":
    main() 