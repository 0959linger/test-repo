"""手动启动河图后端5个llama-server"""
import sys, os, time, subprocess, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_LLAMA_DIR = r"C:\Users\ww109\.qwenpaw\llama.cpp"
_SERVER_EXE = os.path.join(_LLAMA_DIR, "llama-server.exe")
_HOST = "127.0.0.1"

MODELS = {
    "qwen0.5b":   (8080, os.path.join(_LLAMA_DIR, "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf")),
    "qwen1.5b":   (8081, os.path.join(_LLAMA_DIR, "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")),
    "llama1b":    (8082, os.path.join(_LLAMA_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")),
    "smollm135m": (8083, os.path.join(_LLAMA_DIR, "SmolLM2-135M-Instruct-Q4_K_M.gguf")),
    "phi3":       (8084, os.path.join(_LLAMA_DIR, "Phi-3-mini-4k-instruct-Q4_K_M.gguf")),
}

def health(port):
    try:
        req = urllib.request.Request(f"http://{_HOST}:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except: return False

def kill_all():
    subprocess.run(["taskkill", "/f", "/im", "llama-server.exe"], capture_output=True)
    time.sleep(1)

def launch_all():
    kill_all()
    procs = {}
    results = {}
    for name, (port, path) in MODELS.items():
        cmd = [
            _SERVER_EXE, "-m", path, "--port", str(port),
            "--host", _HOST, "--ctx-size", "2048",
            "--n-gpu-layers", "0", "--parallel", "1",
            "--no-webui",
        ]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs[port] = p
        results[name] = {"port": port, "pid": p.pid}
    
    # 等全部就绪
    deadline = time.time() + 60
    ready = set()
    while time.time() < deadline and len(ready) < len(MODELS):
        for name, (port, path) in MODELS.items():
            if port not in ready and health(port):
                ready.add(port)
        if len(ready) < len(MODELS):
            time.sleep(0.3)
    
    for name, (port, path) in MODELS.items():
        status = "UP" if port in ready else "DOWN"
        print(f"  {name:<12} :{port}  {status}")
    
    fails = [name for name, (port, _) in MODELS.items() if port not in ready]
    if fails:
        print(f"\nFAILED: {fails}")
    else:
        print(f"\n全部就绪 ✓")

if __name__ == "__main__":
    launch_all()
