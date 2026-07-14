"""
测试运行器 - 自动启动监控窗口
用法: python run_test_with_monitor.py [test_path]
示例: python run_test_with_monitor.py test_framework/test_engine_v9.py
"""
import subprocess
import sys
import os

def run_with_monitor(test_path="test_framework/"):
    """启动测试监控窗口并运行 pytest"""
    engine_dir = r"C:\Users\ww109\.qwenpaw\workspaces\default\finding-order\engine"
    monitor_script = os.path.join(engine_dir, "test_monitor_auto.py")
    
    # 启动监控窗口（它会自动运行 pytest）
    cmd = [sys.executable, monitor_script, test_path]
    
    print(f"启动测试监控: {' '.join(cmd)}")
    print(f"测试路径: {test_path}")
    print("监控窗口已启动，请查看 GUI 界面...\n")
    
    # 使用 start 命令在新窗口中运行
    subprocess.Popen(
        ["cmd", "/c", "start", "python", monitor_script, test_path],
        cwd=engine_dir
    )
    
    print("✓ 监控窗口已在新窗口启动")

if __name__ == "__main__":
    test_path = sys.argv[1] if len(sys.argv) > 1 else "test_framework/"
    run_with_monitor(test_path)
