"""
启动河图5模型后端（用于真河图全链路测试）

模型配置（来自 hetu/backend.py）：
  qwen0.5b   → 8080
  qwen1.5b   → 8081
  llama1b    → 8082
  smollm135m → 8083
  phi3       → 8084
"""

import subprocess
import time
import requests

LLAMA_DIR = r"C:\Users\ww109\.qwenpaw\llama.cpp"
LLAMA_SERVER = f"{LLAMA_DIR}\\llama-server.exe"

MODELS = {
    "qwen0.5b": {
        "path": f"{LLAMA_DIR}\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "port": 8080,
        "ngl": 0,  # CPU
    },
    "qwen1.5b": {
        "path": f"{LLAMA_DIR}\\Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "port": 8081,
        "ngl": 0,
    },
    "llama1b": {
        "path": f"{LLAMA_DIR}\\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "port": 8082,
        "ngl": 0,
    },
    "smollm135m": {
        "path": f"{LLAMA_DIR}\\SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "port": 8083,
        "ngl": 0,
    },
    "phi3": {
        "path": f"{LLAMA_DIR}\\Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "port": 8084,
        "ngl": 22,  # GPU（phi3 最大，需要GPU加速）
    },
}


def wait_for_server(port, timeout=60):
    """等待服务器启动"""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def start_models():
    """启动所有模型"""
    processes = []
    
    for name, config in MODELS.items():
        print(f"启动 {name} → 端口 {config['port']}...")
        
        cmd = [
            LLAMA_SERVER,
            "-m", config["path"],
            "--port", str(config["port"]),
            "--host", "127.0.0.1",
            "-ngl", str(config["ngl"]),
            "-c", "2048",  # 上下文长度
            "--no-webui",  # 不启动WebUI
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        processes.append((name, proc, config["port"]))
    
    # 等待所有服务器启动
    print("\n等待服务器启动...")
    for name, proc, port in processes:
        if wait_for_server(port):
            print(f"  ✅ {name} (端口 {port})")
        else:
            print(f"  ❌ {name} (端口 {port}) 启动超时")
            proc.kill()
    
    return processes


if __name__ == "__main__":
    print("=" * 60)
    print("河图5模型后端启动器")
    print("=" * 60)
    
    processes = start_models()
    
    print("\n" + "=" * 60)
    print("所有模型已启动，按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止所有模型...")
        for name, proc, port in processes:
            proc.kill()
        print("✅ 已停止")
