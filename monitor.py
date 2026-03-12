#!/usr/bin/env python3
"""PatchCore Training Monitor v2 - Improved Dashboard."""
import json
import subprocess
import re
import os
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

PROJECT_ROOT = Path(os.path.expanduser("~/patchcore"))
OUTPUT_DIR = PROJECT_ROOT / "output"
PROGRESS_FILE = OUTPUT_DIR / "training_progress.json"
PORT = 8585

# All 53 trainable specs (from training order)
ALL_SPECS = [
    "700x300","692x300","594x302","588x300","582x300","428x407","414x405",
    "406x403","400x400","488x300","482x300","440x300","350x357","612x202",
    "350x350","606x201","600x200","596x199","390x300","506x201","500x200",
    "496x199","310x310","W12x8x40","300x305","300x300","450x200","446x199",
    "340x250","200x400","400x200","199x396","396x199","260x256","253x254",
    "250x255","250x250","350x175","298x201","294x200","222x209","300x150",
    "216x206","210x205","244x175","206x204","208x202","203x203","200x200",
    "194x150","150x75","4x34","4x30"
]

def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]|\r', '', text)

def get_gpu_info():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in r.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_util": float(parts[2]),
                    "temp": float(parts[3]),
                    "mem_used": float(parts[4]),
                    "mem_total": float(parts[5]),
                    "power": float(parts[6]),
                })
        return gpus
    except Exception:
        return []

def get_training_process():
    try:
        r = subprocess.run(["pgrep", "-af", "train_v[34]_reorder"], capture_output=True, text=True, timeout=5)
        lines = [l for l in r.stdout.strip().split("\n") if l and "python" in l]
        return len(lines) > 0, lines
    except Exception:
        return False, []

def get_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": []}

def get_log_tail(n=100):
    logpath = None
    for logname in ["train_v4_reorder.log", "train_v3_reorder.log"]:
        path = PROJECT_ROOT / logname
        if path.exists() and path.stat().st_size > 0:
            logpath = path
            break
    if not logpath:
        return [], ""
    # Get meaningful lines (no progress bars)
    try:
        r = subprocess.run(
            f'grep -v "img/s]" {logpath} | grep -v "folder]" | grep -v "batch/s]" | grep -v "batch]" | grep -v "pt]" | grep -v "^$" | grep -v "^\\r" | tail -{n}',
            shell=True, capture_output=True, text=True, timeout=5
        )
        event_lines = [strip_ansi(l).strip() for l in r.stdout.split("\n") if strip_ansi(l).strip() and len(strip_ansi(l).strip()) > 3]
    except Exception:
        event_lines = []
    # Get last progress bar line
    try:
        r2 = subprocess.run(["tail", "-1", str(logpath)], capture_output=True, text=True, timeout=5)
        last = strip_ansi(r2.stdout).strip()
    except Exception:
        last = ""
    return event_lines, last

def parse_current_training(log_lines):
    current = {"spec": None, "group": None, "round": None, "stage": None, "detail": None}
    for line in reversed(log_lines):
        clean = strip_ansi(line).strip()
        if not clean:
            continue
        if "Spec:" in clean and current["spec"] is None:
            m = re.search(r"Spec:\s+(\S+)\s+\|\s+Group\s+(\d+)", clean)
            if m:
                current["spec"] = m.group(1)
                current["group"] = int(m.group(2))
        if "Self-validation round" in clean and current["round"] is None:
            m = re.search(r"round\s+(\d+)/(\d+)", clean)
            if m:
                current["round"] = f"{m.group(1)}/{m.group(2)}"
        if "features:" in clean and current["stage"] is None:
            m = re.search(r"Round\s+(\d+)\s+features:\s+(\d+)%", clean)
            if m:
                current["stage"] = f"Feature extraction (Round {m.group(1)})"
                current["detail"] = f"{m.group(2)}%"
        if "Mask " in clean and current["stage"] is None:
            m = re.search(r"Mask\s+(\S+):\s+(\d+)%", clean)
            if m:
                current["stage"] = "Computing tile masks"
                current["detail"] = f"{m.group(1)} ({m.group(2)}%)"
        if "Scoring:" in clean and current["stage"] is None:
            m = re.search(r"Scoring:\s+(\d+)%", clean)
            if m:
                current["stage"] = "Scoring"
                current["detail"] = f"{m.group(1)}%"
        if "Coreset selection" in clean and current["stage"] is None:
            current["stage"] = "Coreset selection"
        if "Scanning NAS" in clean and current["stage"] is None:
            current["stage"] = "Scanning NAS..."
        if "ONNX" in clean and current["stage"] is None:
            current["stage"] = "ONNX/TRT setup"
        if "Loading backbone" in clean and current["stage"] is None:
            current["stage"] = "Loading backbone..."
    return current

def build_status():
    gpus = get_gpu_info()
    running, procs = get_training_process()
    progress = get_progress()
    log_lines, last_progress = get_log_tail(150)
    current = parse_current_training(log_lines)
    
    completed_set = set(progress.get("completed", []))
    failed_set = set(progress.get("failed", []))
    
    specs_status = []
    for spec in ALL_SPECS:
        groups = []
        for g in range(1, 6):
            key = f"{spec}/group_{g}"
            if key in completed_set:
                status = "done"
            elif key in failed_set:
                status = "failed"
            elif current["spec"] == spec and current["group"] == g:
                status = "current"
            else:
                status = "pending"
            groups.append(status)
        specs_status.append({"spec": spec, "groups": groups})
    
    completed = len(completed_set)
    total = len(ALL_SPECS) * 5
    
    # Clean log: keep meaningful lines + last progress indicator
    display_log = []
    last_progress = None
    for line in log_lines:
        clean = line.strip()
        if not clean:
            continue
        # Track last progress bar but don't add it yet
        if '|' in clean and re.search(r'\d+%', clean):
            last_progress = clean
            continue
        # Skip truncated/partial lines
        if len(clean) < 5:
            continue
        display_log.append(clean)
    # Add last progress bar as current activity
    if last_progress:
        display_log.append("--- Current: " + last_progress)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "gpus": gpus,
        "training": {"running": running, "current": current},
        "progress": {
            "completed": completed, "failed": len(failed_set),
            "total": total, "percent": round(completed / total * 100, 1) if total > 0 else 0,
        },
        "specs": specs_status,
        "log": display_log[-30:],
        "version": "v4 (half-res + TRT + date-stratified)",
    }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PatchCore Monitor</title>
<style>
:root { --bg: #0a0e1a; --card: #131a2e; --border: #1e2a45; --text: #c8d6e5; --dim: #5a6a80; --accent: #3b82f6; --green: #10b981; --red: #ef4444; --yellow: #f59e0b; --purple: #8b5cf6; }
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
.container { max-width: 1200px; margin: 0 auto; padding: 16px; }

/* Header */
.hdr { display:flex; justify-content:space-between; align-items:center; padding:16px 0; border-bottom:1px solid var(--border); margin-bottom:16px; }
.hdr h1 { font-size:20px; color:#fff; display:flex; align-items:center; gap:8px; }
.hdr .ver { font-size:11px; color:var(--dim); }
.badge { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:20px; font-size:12px; font-weight:600; }
.badge.on { background:rgba(16,185,129,.15); color:var(--green); }
.badge.off { background:rgba(239,68,68,.15); color:var(--red); }
.badge .dot { width:8px; height:8px; border-radius:50%; }
.badge.on .dot { background:var(--green); animation:blink 1.5s infinite; }
.badge.off .dot { background:var(--red); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Top cards */
.top { display:grid; grid-template-columns: 1fr 1fr 1.2fr; gap:12px; margin-bottom:16px; }
@media(max-width:900px) { .top { grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:16px; }
.card h3 { font-size:11px; text-transform:uppercase; letter-spacing:1.5px; color:var(--dim); margin-bottom:10px; }
.big { font-size:36px; font-weight:800; color:#fff; }
.sub { font-size:12px; color:var(--dim); margin-top:2px; }
.pbar { background:#1a2340; height:20px; border-radius:10px; overflow:hidden; margin-top:10px; }
.pfill { height:100%; background:linear-gradient(90deg,var(--accent),var(--purple)); border-radius:10px; transition:width .6s; min-width:2px; display:flex; align-items:center; justify-content:center; font-size:10px; color:#fff; font-weight:700; }

/* GPU */
.gpu-row { display:flex; gap:16px; flex-wrap:wrap; }
.gpu-item { flex:1; min-width:100px; }
.gpu-item .label { font-size:11px; color:var(--dim); }
.gpu-item .val { font-size:18px; font-weight:700; color:#fff; }
.minibar { background:#1a2340; height:4px; border-radius:2px; margin-top:3px; overflow:hidden; }
.minibar .fill { height:100%; border-radius:2px; transition:width .5s; }
.minibar .fill.g { background:var(--green); }
.minibar .fill.y { background:var(--yellow); }
.minibar .fill.r { background:var(--red); }

/* Current */
.cur-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.cur-item { background:var(--bg); border-radius:6px; padding:10px; }
.cur-item .cl { font-size:10px; color:var(--dim); text-transform:uppercase; }
.cur-item .cv { font-size:15px; font-weight:600; color:#fff; margin-top:2px; }
.cur-item .cv.highlight { color:var(--yellow); }

/* Spec table */
.spec-card { margin-bottom:16px; }
.spec-table { width:100%; border-collapse:collapse; }
.spec-table th { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--dim); padding:8px 6px; text-align:center; border-bottom:1px solid var(--border); position:sticky; top:0; background:var(--card); z-index:1; }
.spec-table th:first-child { text-align:left; }
.spec-table td { padding:5px 6px; text-align:center; border-bottom:1px solid rgba(30,42,69,.5); font-size:13px; }
.spec-table td:first-child { text-align:left; font-weight:600; font-size:12px; color:var(--text); }
.spec-table tr:hover { background:rgba(59,130,246,.06); }
.spec-table tr.active { background:rgba(245,158,11,.08); }
.g-box { display:inline-block; width:28px; height:22px; border-radius:4px; line-height:22px; font-size:10px; font-weight:700; }
.g-box.done { background:var(--green); color:#000; }
.g-box.failed { background:var(--red); color:#fff; }
.g-box.current { background:var(--yellow); color:#000; animation:blink 1s infinite; }
.g-box.pending { background:#1a2340; color:#3a4560; }
.spec-scroll { max-height:520px; overflow-y:auto; }
.spec-scroll::-webkit-scrollbar { width:6px; }
.spec-scroll::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
.counter { font-size:12px; color:var(--dim); float:right; margin-top:-24px; }

/* Log */
.log-box { background:#050810; border-radius:6px; padding:10px; font-family:'JetBrains Mono','Fira Code',monospace; font-size:11px; line-height:1.7; max-height:250px; overflow-y:auto; color:#6a7a90; white-space:pre-wrap; word-break:break-all; }
.log-box::-webkit-scrollbar { width:4px; }
.log-box::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

.updated { text-align:right; font-size:10px; color:#2a3550; margin-top:8px; }
</style>
</head>
<body>
<div class="container">
  <div class="hdr">
    <div><h1>🔬 PatchCore Training Monitor</h1><div class="ver" id="ver"></div></div>
    <div class="badge" id="badge"><div class="dot"></div><span id="badgeTxt">--</span></div>
  </div>

  <div class="top">
    <div class="card">
      <h3>📊 Progress</h3>
      <div class="big" id="pct">--%</div>
      <div class="sub" id="pctSub">--</div>
      <div class="pbar"><div class="pfill" id="pbar" style="width:0%"></div></div>
    </div>
    <div class="card">
      <h3>🖥️ GPU</h3>
      <div id="gpuArea"><span class="sub">Loading...</span></div>
    </div>
    <div class="card">
      <h3>🔄 Current</h3>
      <div class="cur-grid" id="curArea">
        <div class="cur-item"><div class="cl">Spec</div><div class="cv" id="cSpec">--</div></div>
        <div class="cur-item"><div class="cl">Group</div><div class="cv" id="cGrp">--</div></div>
        <div class="cur-item"><div class="cl">Stage</div><div class="cv highlight" id="cStage">--</div></div>
        <div class="cur-item"><div class="cl">Detail</div><div class="cv" id="cDetail">--</div></div>
      </div>
    </div>
  </div>

  <div class="card spec-card">
    <h3>📦 All Specs (53 specs × 5 groups = 265 models) <span class="counter" id="specCounter"></span></h3>
    <div class="spec-scroll">
      <table class="spec-table">
        <thead><tr><th style="width:140px">Spec</th><th>G1</th><th>G2</th><th>G3</th><th>G4</th><th>G5</th></tr></thead>
        <tbody id="specBody"></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h3>📜 Log</h3>
    <div class="log-box" id="logBox">Loading...</div>
  </div>
  <div class="updated" id="upd"></div>
</div>

<script>
async function refresh() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    
    document.getElementById('ver').textContent = d.version;
    
    const b = document.getElementById('badge');
    const bt = document.getElementById('badgeTxt');
    if (d.training.running) { b.className='badge on'; bt.textContent='Training'; }
    else { b.className='badge off'; bt.textContent='Stopped'; }
    
    document.getElementById('pct').textContent = d.progress.percent + '%';
    document.getElementById('pctSub').textContent = 
      d.progress.completed + ' / ' + d.progress.total + ' models' +
      (d.progress.failed > 0 ? ' (' + d.progress.failed + ' failed)' : '');
    document.getElementById('pbar').style.width = Math.max(d.progress.percent, 0.5) + '%';
    document.getElementById('pbar').textContent = d.progress.percent > 3 ? d.progress.percent + '%' : '';
    
    // GPU
    const ga = document.getElementById('gpuArea');
    if (d.gpus && d.gpus.length > 0) {
      let h = '';
      for (const g of d.gpus) {
        const mp = (g.mem_used/g.mem_total*100).toFixed(0);
        const tp = (g.temp/90*100).toFixed(0);
        h += '<div style="margin-bottom:8px"><div class="sub">GPU'+g.index+': '+g.name+'</div><div class="gpu-row">';
        h += '<div class="gpu-item"><div class="label">Util</div><div class="val">'+g.gpu_util+'%</div><div class="minibar"><div class="fill g" style="width:'+g.gpu_util+'%"></div></div></div>';
        h += '<div class="gpu-item"><div class="label">VRAM</div><div class="val">'+(g.mem_used/1024).toFixed(1)+'/'+(g.mem_total/1024).toFixed(0)+'G</div><div class="minibar"><div class="fill y" style="width:'+mp+'%"></div></div></div>';
        h += '<div class="gpu-item"><div class="label">Temp</div><div class="val">'+g.temp+'°C</div><div class="minibar"><div class="fill r" style="width:'+tp+'%"></div></div></div>';
        h += '<div class="gpu-item"><div class="label">Power</div><div class="val">'+g.power.toFixed(0)+'W</div></div>';
        h += '</div></div>';
      }
      ga.innerHTML = h;
    } else {
      ga.innerHTML = '<span class="sub">GPU data unavailable</span>';
    }
    
    // Current
    const cur = d.training.current;
    document.getElementById('cSpec').textContent = cur.spec || '--';
    document.getElementById('cGrp').textContent = cur.group ? 'Group ' + cur.group : '--';
    document.getElementById('cStage').textContent = cur.stage || '--';
    document.getElementById('cDetail').textContent = cur.detail || '--';
    
    // Spec table
    const tbody = document.getElementById('specBody');
    tbody.innerHTML = '';
    let doneCount = 0;
    for (const s of d.specs) {
      const tr = document.createElement('tr');
      const isCurrent = s.groups.includes('current');
      if (isCurrent) tr.className = 'active';
      
      let cells = '<td>' + s.spec + '</td>';
      for (const g of s.groups) {
        if (g === 'done') doneCount++;
        let label = '';
        if (g === 'done') label = '✓';
        else if (g === 'failed') label = '✗';
        else if (g === 'current') label = '▶';
        cells += '<td><span class="g-box ' + g + '">' + label + '</span></td>';
      }
      tr.innerHTML = cells;
      tbody.appendChild(tr);
    }
    document.getElementById('specCounter').textContent = doneCount + ' / ' + (d.specs.length * 5) + ' done';
    
    // Log
    document.getElementById('logBox').textContent = d.log.join('\n');
    const lb = document.getElementById('logBox');
    lb.scrollTop = lb.scrollHeight;
    
    document.getElementById('upd').textContent = 'Updated: ' + new Date(d.timestamp).toLocaleString('ko-KR');
  } catch(e) {
    document.getElementById('badgeTxt').textContent = 'Error';
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


class MonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            data = json.dumps(build_status(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def main():
    print(f"PatchCore Monitor v2 - http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), MonitorHandler).serve_forever()

if __name__ == "__main__":
    main()
