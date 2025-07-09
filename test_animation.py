#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动画功能的脚本
"""

import os
import sys
import matplotlib
# 使用非交互式后端
matplotlib.use('Agg')

def test_animation_dependencies():
    """测试动画功能所需的依赖"""
    print("=== 测试动画功能依赖 ===")
    
    # 测试matplotlib动画支持
    try:
        import matplotlib.animation as animation
        print("✓ matplotlib.animation 可用")
    except ImportError as e:
        print(f"✗ matplotlib.animation 不可用: {e}")
        return False
    
    # 测试ffmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ ffmpeg 可用")
            ffmpeg_available = True
        else:
            print("✗ ffmpeg 命令执行失败")
            ffmpeg_available = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ ffmpeg 未安装或不在PATH中")
        ffmpeg_available = False
    
    # 测试pillow (用于GIF)
    try:
        from PIL import Image
        print("✓ Pillow 可用 (GIF支持)")
        pillow_available = True
    except ImportError:
        print("✗ Pillow 不可用，无法生成GIF")
        pillow_available = False
    
    return ffmpeg_available or pillow_available

def run_simple_test():
    """运行简单的动画测试"""
    print("\n=== 运行简单动画测试 ===")
    
    try:
        from data_generator import DataGenerator
        from visualizer import ScheduleVisualizer
        from scheduler import OnlineScheduler
        
        # 加载小数据集
        data_loader = DataGenerator()
        vehicles, requests, map_size = data_loader.load_solomon_instance("C101_25.txt", 2)
        
        # 运行调度
        scheduler = OnlineScheduler(vehicles)
        result = scheduler.process_requests(requests)
        
        # 创建可视化器
        visualizer = ScheduleVisualizer(scheduler.vehicles, scheduler.all_requests, map_size)
        
        # 测试简化动画
        print("测试简化动画功能...")
        anim = visualizer._create_simple_animation(save_path="test_simple", fps=2, duration_seconds=5)
        
        if anim is not None:
            print("✓ 简化动画创建成功")
        else:
            print("✗ 简化动画创建失败")
            
        # 测试完整动画（如果有timeline数据）
        if hasattr(scheduler, 'timeline') and scheduler.timeline:
            print("测试完整动画功能...")
            anim2 = visualizer.create_animation_video(
                timeline_data=scheduler.timeline, 
                save_path="test_full", 
                fps=2, 
                duration_seconds=5
            )
            if anim2 is not None:
                print("✓ 完整动画创建成功")
            else:
                print("✗ 完整动画创建失败")
        else:
            print("⚠ 没有timeline数据，跳过完整动画测试")
            
        print("测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("动画功能测试脚本")
    print("=" * 40)
    
    # 检查依赖
    deps_ok = test_animation_dependencies()
    
    if not deps_ok:
        print("\n⚠ 警告: 某些依赖缺失，动画功能可能受限")
        print("建议安装ffmpeg或确保Pillow可用")
    
    # 运行测试
    test_ok = run_simple_test()
    
    if test_ok and deps_ok:
        print("\n🎉 所有测试通过！动画功能可以正常使用")
    elif test_ok:
        print("\n⚠ 基本功能正常，但建议安装完整依赖")
    else:
        print("\n❌ 测试失败，请检查错误信息")

if __name__ == "__main__":
    main() 