# HiSAT: Hierarchical Saliency-Attentive Transformer for Video Summarization

This repository contains the implementation for **HiSAT**, a novel deep learning architecture for automatic video summarization. The core innovation is a **Dual-Path Saliency-Attentive (DPSA) attention mechanism** combined with a **hierarchical temporal pyramid** (HTPE) that models video structure at frame, shot, and scene levels simultaneously.

## Features

- **DPSA Attention**: Integrates visual saliency signals directly into the transformer's attention computation.
- **Hierarchical Temporal Pyramid (HTPE)**: Models video at frame, shot, and scene levels.
- **Saliency-Semantic Fusion Bridge (SSFB)**: Aligns saliency features with semantic representation through cross-attention.
- **Contrastive Redundancy Elimination (CRE)**: Ensures summary diversity using a contrastive loss.
- **Adaptive Summary Budget Predictor**: Predicts optimal video summarization budget based on content.

## Setup

1. Install `uv` if you haven't already (e.g., `curl -LsSf https://astral.sh/uv/install.sh | sh`).

2. Clone the repository and install dependencies using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. Download pre-trained models for feature extraction (GoogLeNet for semantic, TranSalNet for saliency) if extracting custom features.

## Usage

### Training

To train the model on TVSum:
```bash
uv run train.py --config configs/default.yaml
```

### Inference & Web UI

Run the interactive Streamlit summarization app:
```bash
uv run streamlit run app.py
```

Upload an `.mp4` video, and the application will extract features, generate the summary, and visualize scores and shots.

## License

MIT License