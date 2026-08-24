# 🚀 Running Manim Edu Pipeline on Kaggle

Kaggle provides free T4 GPUs (15 GB VRAM) which easily run `qwen2.5:14b` and `maternion/manim-coder` without memory limits or local setup issues.

---

## Quick Start (3 Easy Methods)

### Method 1: Git Clone (Easiest)

1. Open [Kaggle](https://www.kaggle.com/code).
2. Change runtime to GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Open a code cell and clone your pushed repository:
   ```bash
   !git clone https://github.com/PriyadharsanBalaji/EnhancedPipelineReelBased.git
   %cd EnhancedPipelineReelBased
   ```
4. Open `run_kaggle.ipynb` from the Kaggle file browser and run all cells.

### Method 2: Upload Zip & Run

1. **Zip the `EnhancedPipelineReelBased` folder** on your computer.
2. Upload `EnhancedPipelineReelBased.zip` to Kaggle files.
3. Unzip in a code cell:
   ```bash
   !unzip EnhancedPipelineReelBased.zip
   %cd EnhancedPipelineReelBased
   ```
4. Open `run_kaggle.ipynb` or run the setup commands below.

---

## Kaggle Commands Step-by-Step

### 1. Install System Dependencies & LaTeX (for Manim rendering)
```bash
!apt-get update -qq
!apt-get install -y -qq build-essential python3-dev libcairo2-dev libpango1.0-dev ffmpeg zstd texlive-latex-extra texlive-fonts-extra texlive-science tipa
```

### 2. Install Python Libraries
```bash
!pip install manim pymupdf ollama
```

### 3. Install & Start Ollama Server
```bash
!curl -fsSL https://ollama.com/install.sh | sh

import subprocess, time
ollama_process = subprocess.Popen(["ollama", "serve"])
time.sleep(5)
```

### 4. Pull Ollama Models
```bash
!ollama pull maternion/manim-coder
!ollama pull qwen2.5:14b
```

### 5. Run Storyboard & Analysis Only (Review before rendering)
```bash
!python run.py --pdf iemh101.pdf --plan-only
```
To view the generated storyboard in Kaggle:
```python
from IPython.display import Markdown
with open("outputs/iemh101/STORYBOARD.md", "r") as f:
    display(Markdown(f.read()))
```

### 6. Run Full Generation (Code Gen -> Render -> Concatenate)
```bash
!python run.py --pdf iemh101.pdf --resume
```

### 7. Download Outputs
```python
!zip -r outputs_iemh101.zip outputs/iemh101/
from google.colab import files
files.download("outputs_iemh101.zip")
```

---

## 💡 Benefits of Kaggle for this Pipeline
- **15 GB VRAM (T4 GPU)** handles `qwen2.5:14b` and `maternion/manim-coder` effortlessly.
- **Fast rendering**: Manim renders much faster on Kaggle's multi-core CPUs + GPU.
- **Pre-installed utilities**: Linux FFmpeg and LaTeX compile cleanly.
