from flask import Flask, request, jsonify, send_file, render_template_string
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from utils.model import UNetGenerator
import os, io, base64
import numpy as np

app = Flask(__name__)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
OUTPUT_DIR = "outputs/webapp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
def load_latest_model():
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        print("No models folder found - running without model")
        return None
    checkpoints = [f for f in os.listdir("models")
                   if f.startswith("generator") and f.endswith(".pth")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints)[-1]
    model  = UNetGenerator().to(DEVICE)
    model.load_state_dict(torch.load(f"models/{latest}", map_location=DEVICE))
    model.eval()
    print(f"Loaded: {latest}")
    return model

generator = load_latest_model()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def tensor_to_base64(tensor):
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    arr    = (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    img    = Image.fromarray(arr)
    buf    = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def get_gradcam_base64(input_tensor, model):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        class Target:
            def __call__(self, o): return o.abs().mean()

        cam   = GradCAM(model=model, target_layers=[model.enc4[-1]])
        gray  = cam(input_tensor=input_tensor, targets=[Target()])
        img_np = input_tensor[0].permute(1,2,0).cpu().numpy()
        img_np = ((img_np * 0.5 + 0.5) * 255).clip(0,255).astype(np.uint8)
        img_np = img_np.astype(np.float32) / 255.0
        vis   = show_cam_on_image(img_np, gray[0], use_rgb=True)
        buf   = io.BytesIO()
        Image.fromarray(vis).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"GradCAM error: {e}")
        buf = io.BytesIO()
        Image.new('RGB', (256,256), color=(30,30,30)).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaceClean AI</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0a0a;--surface:#141414;--card:#1c1c1c;
  --border:#2a2a2a;--text:#f0f0f0;--muted:#888;
  --accent:#7c6bff;--accent2:#4fd1a5
}
body{font-family:'Segoe UI',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
header{padding:20px 40px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.logo{font-size:20px;font-weight:700;letter-spacing:-0.5px}
.logo span{color:var(--accent)}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{background:#1a1a2e;color:#a89cff;font-size:11px;padding:4px 10px;border-radius:20px;border:1px solid #2d2d50}
.hero{text-align:center;padding:60px 20px 40px}
.hero h1{font-size:42px;font-weight:700;letter-spacing:-1px;line-height:1.1;margin-bottom:12px}
.hero h1 span{color:var(--accent)}
.hero p{color:var(--muted);font-size:16px;max-width:480px;margin:0 auto}
.model-badge{display:inline-flex;align-items:center;gap:6px;background:#0a1a0a;color:var(--accent2);font-size:12px;padding:6px 12px;border-radius:20px;border:1px solid #1a3a1a;margin-top:16px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent2)}
.main{max-width:900px;margin:0 auto;padding:0 20px 60px}
.upload-zone{border:2px dashed var(--border);border-radius:16px;padding:48px;text-align:center;cursor:pointer;transition:all 0.2s;background:var(--surface);margin-bottom:24px}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:#0f0f1a}
.upload-icon{width:48px;height:48px;margin:0 auto 16px;opacity:0.4;display:block}
.upload-zone p{color:var(--muted);font-size:14px;margin-top:8px}
.upload-zone strong{color:var(--text)}
#file-input{display:none}
.preview-thumb{width:80px;height:80px;border-radius:8px;object-fit:cover;margin:12px auto 0;display:none;border:2px solid var(--accent)}
.btn{background:var(--accent);color:#fff;border:none;padding:14px 32px;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer;transition:all 0.2s;width:100%}
.btn:hover{background:#6b5ce7;transform:translateY(-1px)}
.btn:disabled{background:#333;color:#666;cursor:not-allowed;transform:none}
.progress{height:4px;background:var(--border);border-radius:2px;margin:16px 0;overflow:hidden;display:none}
.progress-bar{height:100%;background:var(--accent);border-radius:2px;width:0;transition:width 0.3s}
.results{display:none;margin-top:32px}
.results-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.result-card{background:var(--card);border-radius:12px;overflow:hidden;border:1px solid var(--border)}
.result-card img{width:100%;display:block;aspect-ratio:1;object-fit:cover;background:#111}
.result-label{padding:12px 16px;font-size:13px;color:var(--muted);display:flex;align-items:center;justify-content:space-between}
.result-label .pill{background:var(--surface);padding:3px 8px;border-radius:6px;font-size:11px}
.gradcam-card{background:var(--card);border-radius:12px;overflow:hidden;border:1px solid var(--border);margin-bottom:16px}
.gradcam-card img{width:100%;display:block}
.gradcam-label{padding:12px 16px;font-size:13px;color:var(--muted)}
.gradcam-label strong{color:var(--accent2)}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.stat{background:var(--card);border-radius:10px;padding:16px;text-align:center;border:1px solid var(--border)}
.stat-value{font-size:22px;font-weight:700;color:var(--accent)}
.stat-label{font-size:12px;color:var(--muted);margin-top:4px}
.download-row{display:flex;gap:12px}
.btn-outline{background:transparent;color:var(--text);border:1px solid var(--border);padding:10px 20px;border-radius:8px;font-size:13px;font-weight:500;cursor:pointer;transition:all 0.2s;flex:1}
.btn-outline:hover{border-color:var(--accent);color:var(--accent)}
.error{background:#1a0a0a;border:1px solid #3a1a1a;color:#f87171;padding:12px 16px;border-radius:8px;font-size:14px;margin-top:12px;display:none}
footer{text-align:center;padding:24px;border-top:1px solid var(--border);color:var(--muted);font-size:13px}
</style>
</head>
<body>

<header>
  <div class="logo">Face<span>Clean</span> AI</div>
  <div class="badges">
    <div class="badge">Attention GAN</div>
    <div class="badge">FaceNet Identity</div>
    <div class="badge">GradCAM XAI</div>
    <div class="badge">CelebA trained</div>
  </div>
</header>

<div class="hero">
  <h1>Remove facial hair<br>with <span>AI precision</span></h1>
  <p>Powered by Attention U-Net GAN with identity preservation — trained on CelebA dataset</p>
  <div class="model-badge">
    <div class="dot"></div>
    Model active &mdash; {{ device }}
  </div>
</div>

<div class="main">

  <div class="upload-zone" id="drop-zone"
       onclick="document.getElementById('file-input').click()">
    <svg class="upload-icon" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.5">
      <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
    </svg>
    <strong>Drop a face photo here</strong>
    <p>or click to browse &mdash; JPG, PNG supported</p>
    <input type="file" id="file-input" accept="image/*"
           onchange="handleFile(this.files[0])">
    <img id="preview-thumb" class="preview-thumb" src="" alt="preview">
  </div>

  <button class="btn" id="submit-btn" onclick="submitImage()" disabled>
    Remove facial hair
  </button>

  <div class="progress" id="progress">
    <div class="progress-bar" id="progress-bar"></div>
  </div>
  <div class="error" id="error-box"></div>

  <div class="results" id="results">

    <div class="results-grid">
      <div class="result-card">
        <img id="input-img" src="" alt="Input">
        <div class="result-label">
          Original <span class="pill">with beard</span>
        </div>
      </div>
      <div class="result-card">
        <img id="output-img" src="" alt="Output">
        <div class="result-label">
          Generated <span class="pill">beard removed</span>
        </div>
      </div>
    </div>

    <div class="gradcam-card">
      <img id="gradcam-img" src="" alt="GradCAM Heatmap">
      <div class="gradcam-label">
        <strong>GradCAM heatmap</strong>
        &mdash; red/warm regions show where the model focused to remove beard
      </div>
    </div>

    <div class="stats">
      <div class="stat">
        <div class="stat-value" id="stat-time">—</div>
        <div class="stat-label">Processing time</div>
      </div>
      <div class="stat">
        <div class="stat-value">256px</div>
        <div class="stat-label">Resolution</div>
      </div>
      <div class="stat">
        <div class="stat-value" id="stat-device">—</div>
        <div class="stat-label">Hardware</div>
      </div>
    </div>

    <div class="download-row">
      <button class="btn-outline"
              onclick="downloadImg('output-img','faceclean_result.png')">
        Download result
      </button>
      <button class="btn-outline"
              onclick="downloadImg('gradcam-img','faceclean_gradcam.png')">
        Download heatmap
      </button>
    </div>

  </div>
</div>

<footer>
  Built with PyTorch &middot; Attention U-Net &middot; FaceNet &middot; Flask
  &nbsp;|&nbsp; Facial Hair Occlusion Removal &mdash; Final Year Project
</footer>

<script>
let selectedFile = null;
const dropZone   = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    const thumb = document.getElementById('preview-thumb');
    thumb.src   = e.target.result;
    thumb.style.display = 'block';
    document.getElementById('input-img').src = e.target.result;
    document.getElementById('submit-btn').disabled = false;
    document.getElementById('output-img').src   = '';
    document.getElementById('gradcam-img').src  = '';
    document.getElementById('results').style.display = 'none';
  };
  reader.readAsDataURL(file);
}

function animateProgress() {
  const bar = document.getElementById('progress-bar');
  let w = 0;
  return setInterval(() => {
    w = Math.min(w + Math.random() * 8, 90);
    bar.style.width = w + '%';
  }, 200);
}

async function submitImage() {
  if (!selectedFile) return;
  const btn      = document.getElementById('submit-btn');
  const progress = document.getElementById('progress');
  const errorBox = document.getElementById('error-box');

  btn.disabled        = true;
  btn.textContent     = 'Processing...';
  progress.style.display = 'block';
  errorBox.style.display = 'none';

  const start    = Date.now();
  const interval = animateProgress();
  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/api/remove', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('output-img').src  = 'data:image/png;base64,' + data.output;
    document.getElementById('gradcam-img').src = 'data:image/png;base64,' + data.gradcam;
    document.getElementById('stat-time').textContent   = ((Date.now()-start)/1000).toFixed(1) + 's';
    document.getElementById('stat-device').textContent = data.device.toUpperCase();
    document.getElementById('results').style.display   = 'block';

  } catch(e) {
    errorBox.textContent     = 'Error: ' + e.message;
    errorBox.style.display   = 'block';
  } finally {
    clearInterval(interval);
    btn.disabled    = false;
    btn.textContent = 'Remove facial hair';
    setTimeout(() => {
      progress.style.display = 'none';
      document.getElementById('progress-bar').style.width = '0';
    }, 500);
  }
}

function downloadImg(id, filename) {
  const a  = document.createElement('a');
  a.href   = document.getElementById(id).src;
  a.download = filename;
  a.click();
}
</script>
</body>
</html>"""

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML, device=DEVICE)

@app.route("/api/remove", methods=["POST"])
def api_remove():
    if generator is None:
        return jsonify({"error": "No model found! Train first."})
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded!"})
    try:
        file         = request.files["image"]
        img          = Image.open(file.stream).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = generator(input_tensor)

        output_b64  = tensor_to_base64(output)
        gradcam_b64 = get_gradcam_base64(input_tensor, generator)

        return jsonify({
            "output":  output_b64,
            "gradcam": gradcam_b64,
            "device":  DEVICE
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  generator is not None,
        "device": DEVICE
    })

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("Starting FaceClean AI at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)