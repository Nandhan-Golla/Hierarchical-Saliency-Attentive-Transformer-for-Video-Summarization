import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pandas as pd
import json
import os

from utils.kts import kts_segmentation
from utils.knapsack import knapsack_ortools
from utils.assembly import assemble_summary
from data.video_utils import extract_frames
from data.extract_features import FeatureExtractor
from models.hisat import HiSAT

# Set up page config
st.set_page_config(page_title="HiSAT Video Summarization", page_icon="🎬", layout="wide")

st.title("🎬 HiSAT Video Summarizer")
st.markdown("Upload a video and our Hierarchical Saliency-Attentive Transformer will generate a dense summary based on adaptive attention and predictive saliency.")

st.sidebar.header("Settings")
budget_pct = st.sidebar.slider("Summary Budget (%)", min_value=5, max_value=50, value=15, step=5)
model_choice = st.sidebar.selectbox("Model", ["HiSAT (Recommended)", "Base Transformer"])

uploaded_file = st.file_uploader("📁 UPLOAD VIDEO (Supported: .mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = FeatureExtractor(device=device)
    # Initialize un-trained dummy weights for demonstration, in a real scenario we'd load weights here
    hisat = HiSAT().to(device)
    hisat.eval()
    return extractor, hisat, device

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("🚀 Generate Summary"):
        with st.spinner("Initializing models..."):
            extractor, model, device = load_models()
        
        with st.spinner("Extracting frames and finding shot boundaries..."):
            frames_tensor, original_frames = extract_frames(video_path)
            N = frames_tensor.size(0)
            
            # Use basic uniform boundaries since dynamic takes time 
            shot_boundaries = list(range(0, N, max(1, N // 10)))
            
        with st.spinner("Extracting semantic and saliency features..."):
            F_sem = extractor.extract_semantic(frames_tensor)
            F_sal_spatial, s_scores = extractor.extract_saliency(frames_tensor)
            
            # Add batch dimension
            F_sem = F_sem.unsqueeze(0).to(device)
            F_sal_spatial = F_sal_spatial.unsqueeze(0).to(device)
            s_scores = s_scores.unsqueeze(0).to(device)
            
        with st.spinner("Running HiSAT Inference..."):
            with torch.no_grad():
                pred_scores, pred_budget, h_temporal = model(F_sem, F_sal_spatial, s_scores, shot_boundaries)
                
            scores_np = pred_scores[0].cpu().numpy()
            
            # Apply KTS and Knapsack
            kts_segments = kts_segmentation(h_temporal[0].cpu().numpy())
            
            # Aggregate score per segment
            seg_scores = []
            seg_lengths = []
            for start, end in kts_segments:
                if end > start:
                    seg_scores.append(float(np.mean(scores_np[start:end])))
                    seg_lengths.append(end - start)
                else:
                    seg_scores.append(0.0)
                    seg_lengths.append(0)
                    
            budget_frames = int(N * (budget_pct / 100.0))
            selected_indices = knapsack_ortools(seg_lengths, seg_scores, budget_frames)
            
        with st.spinner("Assembling final video..."):
            out_video_path, out_scores_path = assemble_summary(video_path, scores_np, kts_segments, selected_indices)
            
        st.success("Summary Generated!")
        
        st.subheader("📊 Frame Importance Scores")
        # Ensure scores length matches length of a range index
        df = pd.DataFrame({"Frame": np.arange(len(scores_np)), "Importance Score": scores_np})
        st.area_chart(df.set_index("Frame"))
        
        st.subheader("🎥 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Video**")
            st.video(video_path)
        with col2:
            st.markdown("**Summarized Video**")
            st.video(out_video_path)
            
        st.markdown("---")
        
        with open(out_scores_path, "rb") as f:
            st.download_button("📄 Download Scores JSON", f, file_name="scores.json", mime="application/json")
            
        with open(out_video_path, "rb") as fp:
            st.download_button("💾 Download Summary Video", fp, file_name="summary_video.mp4", mime="video/mp4")

        # Cleanup
        os.remove(tfile.name)
