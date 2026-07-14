"""
八卦构架2 测试面板 — 零依赖 Web 前端
用法: python run_web.py
打开 http://127.0.0.1:8099
"""
import http.server, threading, json, time, queue, os, sys, io, numpy as np

PORT = 8099
ROOT = r'C:\Users\ww109\.qwenpaw\workspaces\default'
ENGINE = os.path.join(ROOT, 'finding-order', 'engine')
HETU = os.path.join(ROOT, 'hetu')

sys.path.insert(0, ENGINE)
sys.path.insert(0, HETU)
sys.path.insert(0, os.path.join(ROOT, 'hetu'))

from engine_v9 import EngineV9, BAGUA
from breathing_v38 import once
from memory_layer import MemoryLayer as HetuMemoryLayer

# ─── 全局状态 ───
progress_q = queue.Queue()  # 进度事件队列
results = []
current_state = {"status": "idle", "total": 0, "done": 0, "results": []}

TESTS = [
    ('我升职了！', '升职'),
    ('朋友背叛了我，我的心被刀割一样。', '背叛'),
    ('有人偷了我的钱包。', '偷窃'),
    ('你怎么定义美？', '美'),
    ('爱到底是什么？', '爱'),
    ('计算：1+1等于几？', '数学'),
    ('如果这句话是假的，那它是真的吗？', '说谎者'),
    ('进化论正确吗？', '进化论'),
]


def run_tests():
    """后台跑测试，推送进度到队列"""
    global current_state
    current_state = {"status": "running", "total": len(TESTS), "done": 0, "results": []}
    progress_q.put(current_state.copy())

    engine = EngineV9(hour=12,
        embed_path=os.path.join(ROOT, 'finding-order', 'data', 'qwen7b_embed_tokens.npy'),
        pca_path=os.path.join(ROOT, 'finding-order', 'data', 'pca_256_proj.npz'))
    memory = HetuMemoryLayer()

    try:
        for text, label in TESTS:
            t0 = time.time()
            r = once(question=text, memory=memory,
                steam_url='http://127.0.0.1:8080', trace=False, reason_c=True, near_window=None)
            ht = [e.text for e in r['hetu'].final_8[:6]]
            qf = np.array([r['qi'].get(g, 1.0/8) for g in BAGUA])
            r2 = engine.perceive(text=text, hetu_texts=ht, qi_field=qf)
            elapsed = time.time() - t0

            row = {
                "label": label,
                "hetu_cv": round(r['cv'], 2),
                "hetu_top": r['top'],
                "v9_cv": round(r2.cv, 2),
                "v9_winner": r2.winner,
                "time": round(elapsed, 1),
            }
            current_state["results"].append(row)
            current_state["done"] += 1
            progress_q.put(current_state.copy())

        current_state["status"] = "done"
    except Exception as e:
        current_state["status"] = "error"
        current_state["error"] = str(e)
    progress_q.put(current_state.copy())


# ─── HTML 页面 ───
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>八卦构架2 测试面板</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Microsoft YaHei',sans-serif;background:#0a0a0f;color:#ddd;padding:20px;min-height:100vh}
  h1{color:#e8c170;text-align:center;margin-bottom:5px;font-size:24px}
  .sub{text-align:center;color:#888;margin-bottom:20px;font-size:12px}
  .controls{text-align:center;margin-bottom:20px}
  button{padding:10px 30px;font-size:16px;border:none;border-radius:6px;cursor:pointer;margin:0 5px}
  .btn-start{background:#2a7d2a;color:#fff}
  .btn-start:hover{background:#3a9d3a}
  .btn-start:disabled{background:#444;cursor:not-allowed}
  .progress-box{background:#111;border-radius:8px;padding:15px;margin-bottom:20px;max-width:700px;margin-left:auto;margin-right:auto}
  .progress-bar{background:#222;border-radius:10px;height:24px;overflow:hidden;margin:10px 0}
  .progress-fill{background:linear-gradient(90deg,#2a7d2a,#4ae54a);height:100%;width:0;transition:width .3s;border-radius:10px}
  .progress-text{text-align:center;font-size:14px}
  table{width:100%;max-width:700px;margin:0 auto;border-collapse:collapse}
  th{background:#1a1a2e;color:#e8c170;padding:8px;text-align:left;font-size:13px;border-bottom:1px solid #333}
  td{padding:7px 8px;font-size:13px;border-bottom:1px solid #1a1a2e}
  tr:hover{background:#111122}
  .pass{color:#4ae54a}
  .status{text-align:center;padding:20px;font-size:18px}
  .status.done{color:#4ae54a}
  .status.error{color:#ff4444}
  .hetu{color:#e8c170}
  .v9{color:#6bc}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
  .running-indicator{animation:pulse 1s infinite;color:#e8c170;text-align:center;font-size:14px;margin:10px 0}
</style>
</head>
<body>
<h1>☯ 八卦构架2 测试面板</h1>
<div class="sub">真河图5模型全链路 — 实时进度</div>

<div class="controls">
  <button class="btn-start" id="btnStart" onclick="startTest()">▶ 开始测试</button>
</div>

<div class="progress-box">
  <div class="progress-text" id="progressText">就绪，等待开始...</div>
  <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
  <div id="runningInd" style="display:none" class="running-indicator">● 运行中...</div>
</div>

<table id="resultTable">
  <thead><tr><th>题</th><th>河图CV</th><th>河图卦</th><th>v9 CV</th><th>v9 卦</th><th>耗时</th></tr></thead>
  <tbody></tbody>
</table>
<div id="finalStatus" class="status"></div>

<script>
let eventSource = null;

function startTest(){
  document.getElementById('btnStart').disabled = true;
  document.getElementById('runningInd').style.display = 'block';
  document.getElementById('progressText').textContent = '启动中...';

  fetch('/api/start', {method:'POST'}).then(r => r.json()).then(data => {
    if(data.ok) listenProgress();
  });
}

function listenProgress(){
  if(eventSource) eventSource.close();
  eventSource = new EventSource('/api/progress');

  eventSource.onmessage = function(e){
    let state = JSON.parse(e.data);
    let pct = state.total > 0 ? (state.done / state.total * 100) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressText').textContent = state.done + ' / ' + state.total;

    // 更新表格
    let tbody = document.querySelector('#resultTable tbody');
    tbody.innerHTML = state.results.map(r =>
      '<tr><td>'+r.label+'</td><td class="hetu">'+r.hetu_cv+'</td><td class="hetu">'+r.hetu_top+'</td><td class="v9">'+r.v9_cv+'</td><td class="v9">'+r.v9_winner+'</td><td>'+r.time+'s</td></tr>'
    ).join('');

    if(state.status === 'done'){
      document.getElementById('runningInd').style.display = 'none';
      document.getElementById('btnStart').disabled = false;
      document.getElementById('finalStatus').className = 'status done';
      document.getElementById('finalStatus').textContent = '✅ 全部完成! ' + state.done + '/' + state.total + ' 通过';
      eventSource.close();
    } else if(state.status === 'error'){
      document.getElementById('runningInd').style.display = 'none';
      document.getElementById('btnStart').disabled = false;
      document.getElementById('finalStatus').className = 'status error';
      document.getElementById('finalStatus').textContent = '❌ 出错: ' + (state.error || '未知');
      eventSource.close();
    }
  };

  eventSource.onerror = function(){
    document.getElementById('runningInd').style.display = 'none';
    document.getElementById('btnStart').disabled = false;
  };
}
</script>
</body>
</html>"""


# ─── HTTP 服务器 ───
class TestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
        elif self.path == '/api/progress':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            last_done = -1
            while True:
                try:
                    state = progress_q.get(timeout=0.5)
                    data = f"data: {json.dumps(state, ensure_ascii=False)}\n\n"
                    self.wfile.write(data.encode('utf-8'))
                    self.wfile.flush()
                    last_done = state.get('done', last_done)
                    if state.get('status') in ('done', 'error'):
                        break
                except queue.Empty:
                    # 心跳
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    continue
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/start':
            if current_state.get('status') == 'running':
                self.send_json({"ok": False, "msg": "正在运行"})
            else:
                threading.Thread(target=run_tests, daemon=True).start()
                self.send_json({"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


if __name__ == '__main__':
    print(f"\n  ☯ 八卦构架2 测试面板")
    print(f"  打开浏览器: http://127.0.0.1:{PORT}\n")
    httpd = http.server.HTTPServer(('127.0.0.1', PORT), TestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n关闭")
        httpd.shutdown()
