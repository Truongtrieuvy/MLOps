from flask import Flask, request, jsonify, render_template_string
import mlflow.sklearn, numpy as np, os
from pathlib import Path

# ====== Load model từ thư mục app/model (hoặc MODEL_DIR nếu có) ======
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "model"))

model = mlflow.sklearn.load_model(MODEL_DIR)
N = int(model.n_features_in_)

app = Flask(__name__)

# ====== HTML template đơn giản nhưng gọn & đẹp (Tailwind-lite tự viết) ======
TPL = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>MLOps Demo 1 – Flask UI</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0f172a;
      --card: #020617;
      --accent: #22c55e;
      --accent-soft: #22c55e22;
      --border: #1e293b;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --danger: #ef4444;
      --danger-soft: #fecaca33;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1f2937 0, #020617 55%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .shell {
      width: 100%;
      max-width: 960px;
      background: linear-gradient(135deg, #020617dd, #020617f5);
      border-radius: 24px;
      border: 1px solid #1f2937;
      box-shadow:
        0 30px 80px rgba(0,0,0,.65),
        0 0 0 1px rgba(148,163,184,.15);
      padding: 24px 26px 22px;
    }
    @media (min-width: 768px) {
      .shell { padding: 26px 32px 26px; }
      .grid { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2.2fr); gap: 18px; }
    }
    .row { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    h1 {
      font-size: 24px;
      font-weight: 650;
      letter-spacing: -0.03em;
      display:flex;
      align-items:center;
      gap:8px;
      margin: 0 0 2px;
    }
    h1 span.logo {
      width: 26px;
      height: 26px;
      border-radius: 999px;
      background: radial-gradient(circle at 30% 30%, #a5b4fc, #4f46e5);
      display:flex;
      align-items:center;
      justify-content:center;
      font-size: 15px;
      color:white;
      box-shadow: 0 0 0 1px rgba(129,140,248,.65), 0 0 20px rgba(129,140,248,.8);
    }
    .tag {
      display:inline-flex;
      align-items:center;
      gap:4px;
      padding:3px 9px;
      border-radius:999px;
      border:1px solid #1e293b;
      background:#020617;
      color:var(--muted);
      font-size:11px;
      text-transform:uppercase;
      letter-spacing:.09em;
    }
    .tag-dot {
      width:7px;height:7px;border-radius:999px;background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.35);
    }
    p.sub {
      margin:4px 0 14px;
      font-size: 13px;
      color: var(--muted);
    }
    textarea {
      resize: vertical;
      width: 100%;
      min-height: 120px;
      max-height: 220px;
      padding: 11px 12px;
      border-radius: 14px;
      border: 1px solid #1f2937;
      background: #020617;
      color: var(--text);
      font-size: 13px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      line-height: 1.5;
      box-shadow: inset 0 0 0 1px rgba(15,23,42,.25);
    }
    textarea:focus {
      outline: none;
      border-color: #6366f1;
      box-shadow:
        0 0 0 1px #4f46e5,
        0 0 0 4px rgba(79,70,229,.35);
    }
    label { font-size:13px; font-weight:500; color:#e5e7eb; display:block; margin-bottom:6px; }
    .muted { font-size:12px; color:var(--muted); margin-top:4px;}
    button {
      border:none;
      border-radius:999px;
      padding:8px 16px;
      font-size:13px;
      font-weight:500;
      cursor:pointer;
      display:inline-flex;
      align-items:center;
      gap:6px;
      transition: all .15s ease;
      white-space:nowrap;
    }
    .btn-primary {
      background: linear-gradient(135deg,#22c55e,#16a34a);
      color:#04101a;
      box-shadow: 0 12px 30px rgba(34,197,94,.4);
    }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 16px 40px rgba(34,197,94,.55); }
    .btn-ghost {
      background: rgba(15,23,42,.8);
      color: var(--muted);
      border: 1px solid #1e293b;
    }
    .btn-ghost:hover { border-color:#4b5563; color:#e5e7eb; background:#020617; }
    .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
    .chip {
      font-size:11px;
      padding:3px 7px;
      border-radius:999px;
      border:1px solid #111827;
      background:#020617;
      color:#9ca3af;
    }
    .metric-card {
      border-radius:16px;
      border:1px solid #1f2937;
      background: radial-gradient(circle at top left,#111827,#020617);
      padding:12px 14px 11px;
      margin-top:10px;
    }
    .metric-label { font-size:11px; text-transform:uppercase; letter-spacing:.12em; color:#9ca3af; margin-bottom:4px;}
    .metric-value { font-size:22px; font-weight:650; letter-spacing:-0.03em;}
    .metric-pill {
      display:inline-flex;align-items:center;gap:6px;
      font-size:12px;color:var(--muted);margin-top:6px;
    }
    .dot-pos {width:8px;height:8px;border-radius:999px;background:var(--accent);box-shadow:0 0 0 4px var(--accent-soft);}
    .dot-neg {width:8px;height:8px;border-radius:999px;background:var(--danger);box-shadow:0 0 0 4px var(--danger-soft);}
    .badge {
      display:inline-flex;
      align-items:center;
      gap:5px;
      padding:3px 8px;
      border-radius:999px;
      background:#0b1220;
      border:1px solid #1e293b;
      font-size:11px;
      color:var(--muted);
      margin-left:6px;
    }
    .badge span { font-weight:500; color:#e5e7eb; }
    .explain {
      margin-top:8px;
      font-size:12px;
      color:var(--muted);
    }
    .explain ul { margin:5px 0 0 18px; padding:0; }
    .explain li { margin:2px 0; }
    .footer {
      margin-top:12px;
      font-size:11px;
      color:#6b7280;
      display:flex;
      justify-content:space-between;
      gap:10px;
      flex-wrap:wrap;
    }
    .error-box {
      margin-top:10px;
      padding:8px 10px;
      border-radius:12px;
      border:1px solid #7f1d1d;
      background:#450a0a;
      color:#fecaca;
      font-size:12px;
    }
    .pill {
      font-size:11px;
      padding:3px 8px;
      border-radius:999px;
      background:#020617;
      border:1px solid #1e293b;
      color:#9ca3af;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="row">
      <div>
        <h1>
          <span class="logo">ML</span>
          MLOps Flask Web UI
        </h1>
        <p class="sub">
          Demo dự đoán phân loại sử dụng mô hình tốt nhất đã lưu bằng MLflow.
          <span class="badge">n_features: <span>{{ n_features }}</span></span>
        </p>
      </div>
      <div class="tag">
        <span class="tag-dot"></span>
        Production · Docker · {{ model_name }}
      </div>
    </div>

    <div class="grid">
      <div>
        <form method="POST">
          <label for="features">Nhập {{ n_features }} đặc trưng (cách nhau bằng dấu phẩy):</label>
          <textarea id="features" name="features"
            placeholder="0.12, -0.03, 1.5, ... (đủ {{ n_features }} số)">{{ features_text }}</textarea>
          <p class="muted">
            Gợi ý: bạn có thể bấm <b>Random hợp lệ</b> để tự sinh đúng {{ n_features }} đặc trưng.
          </p>
          <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
            <button class="btn-primary" type="submit">✨ Dự đoán</button>
            <button class="btn-ghost" type="button" onclick="document.getElementById('randForm').submit();">
              🎲 Random hợp lệ
            </button>
          </div>
        </form>

        <form id="randForm" method="POST" style="display:none;">
          <input type="hidden" name="random" value="1" />
        </form>

        <div class="chips">
          <span class="chip">Flask</span>
          <span class="chip">MLflow model registry</span>
          <span class="chip">Docker image trên Docker Hub</span>
          <span class="chip">Môn Machine Learning DevOps</span>
        </div>
      </div>

      <div>
        <div class="metric-card">
          <div class="metric-label">Kết quả dự đoán</div>
          {% if error %}
            <div class="error-box">
              ⚠️ {{ error }}
            </div>
          {% else %}
            <div class="metric-value">
              {% if prediction is not none %}
                {{ prediction }}
              {% else %}
                —
              {% endif %}
            </div>
            {% if prob is not none %}
              <div class="metric-pill">
                {% if prediction == 1 %}
                  <span class="dot-pos"></span>
                  <span>Positive (lớp 1) · xác suất ≈ {{ "%.3f"|format(prob) }}</span>
                {% elif prediction == 0 %}
                  <span class="dot-neg"></span>
                  <span>Negative (lớp 0) · xác suất lớp 1 ≈ {{ "%.3f"|format(prob) }}</span>
                {% endif %}
              </div>
            {% else %}
              <div class="metric-pill">
                <span class="dot-pos"></span>
                <span>Mô hình không xuất xác suất (predict_proba không có).</span>
              </div>
            {% endif %}
          {% endif %}

          <div class="explain">
            <strong>Giải thích nhanh để thuyết trình:</strong>
            <ul>
              <li>Mô hình nhận vào <b>{{ n_features }}</b> đặc trưng dạng số.</li>
              <li>Flask nhận input &rarr; chuyển thành vector NumPy &rarr; gọi model.</li>
              <li>Kết quả &amp; xác suất được hiển thị ngay trên giao diện web.</li>
            </ul>
          </div>
        </div>

        <div class="footer">
          <span>✅ Đã load model từ <code>{{ model_path }}</code></span>
          <span>
            <span class="pill">/schema</span>
            <span class="pill">/predict (JSON API)</span>
          </span>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""

# ====== Logic dự đoán dùng chung cho API & Web ======
def _predict_core(features_list):
    x = np.array(features_list, dtype=float).reshape(1, -1)
    if x.shape[1] != N:
        raise ValueError(f"Cần {N} đặc trưng, nhưng nhận được {x.shape[1]}")
    y = int(model.predict(x)[0])
    prob = float(model.predict_proba(x)[0][-1]) if hasattr(model, "predict_proba") else None
    return y, prob


# ====== Web UI đẹp ở "/" (GET + POST) ======
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    prob = None
    features_text = ""

    # Nếu bấm Random -> sinh input hợp lệ
    if request.method == "POST" and request.form.get("random") == "1":
        sample = np.random.randn(N).round(4).tolist()
        features_text = ", ".join(map(str, sample))
    elif request.method == "POST":
        raw = request.form.get("features", "").strip()
        features_text = raw
        try:
            arr = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
            y, p = _predict_core(arr)
            prediction, prob = y, p
        except ValueError as e:
            error = str(e)
        except Exception:
            error = "Đầu vào không hợp lệ. Vui lòng nhập chuỗi số thực cách nhau bởi dấu phẩy."

    return render_template_string(
        TPL,
        n_features=N,
        prediction=prediction,
        prob=prob,
        error=error,
        features_text=features_text,
        model_name=type(model).__name__,
        model_path=MODEL_DIR,
    )


# ====== API JSON vẫn giữ để demo DevOps ======
@app.get("/schema")
def schema():
    return jsonify({"n_features": N, "model": type(model).__name__, "model_path": MODEL_DIR})


@app.post("/predict")
def predict_api():
    data = request.get_json(force=True)
    try:
        arr = data["features"]
        y, p = _predict_core(arr)
        return jsonify({"prediction": y, "prob_pos": p})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
