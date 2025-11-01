#!/usr/bin/env python3
"""
Streamlit Demo App for CheXpert Label Prediction with CheXagent

Replaces LLaVA-based app_demo.py with CheXagent-based inference.
Uses results from the 5k pipeline evaluation.
"""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from PIL import Image
import os

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Page config
st.set_page_config(
    page_title="CheXpert Prediction - CheXagent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

CHEXPERT13 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
CHEXPERT14 = CHEXPERT13 + ["No Finding"]


@st.cache_data
def load_results_with_impressions(results_csv: Path):
    """Load results CSV with impressions."""
    if not results_csv.exists():
        return None
    df = pd.read_csv(results_csv)
    return df


@st.cache_data
def load_sample_images(image_list_txt: Path, max_samples: int = 50):
    """Load sample image paths."""
    if not image_list_txt.exists():
        return []
    
    images = []
    with open(image_list_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line and Path(line).exists():
                images.append(line)
                if len(images) >= max_samples:
                    break
    return images


def display_chexpert_labels(labels_dict, title: str = "CheXpert Labels"):
    """Display CheXpert labels in a formatted way."""
    st.markdown(f"**{title}:**")
    
    positives = [label for label, val in labels_dict.items() if val == 1]
    negatives = [label for label, val in labels_dict.items() if val == 0 and label != "No Finding"]
    
    if positives:
        st.markdown("✅ **Positive:**")
        for label in positives:
            st.markdown(f"  - {label}")
    
    if negatives:
        st.markdown("❌ **Negative:**")
        for label in negatives[:5]:  # Show first 5
            st.markdown(f"  - {label}")
        if len(negatives) > 5:
            st.markdown(f"  ... and {len(negatives) - 5} more")


def main():
    st.markdown('<h1 class="main-header">🏥 CheXpert Label Prediction - CheXagent</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Demo**: View predictions and impressions from 5k evaluation results")
    
    project_root = Path(__file__).parent
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Load results
    results_path = project_root / "outputs_5k" / "final" / "test_with_impressions.csv"
    if not results_path.exists():
        st.error(f"❌ Results file not found: {results_path}")
        st.info("Run the 5k pipeline first to generate results.")
        return
    
    results_df = load_results_with_impressions(results_path)
    if results_df is None:
        st.error("Failed to load results")
        return
    
    st.sidebar.info(f"📊 **Total Samples**: {len(results_df)}")
    
    # Sample selection
    st.sidebar.subheader("Select Sample")
    sample_idx = st.sidebar.selectbox(
        "Sample Index",
        range(len(results_df)),
        format_func=lambda x: f"Sample {x+1}: {Path(results_df.iloc[x].get('image', '')).name if 'image' in results_df.columns else f'Row {x+1}'}"
    )
    
    selected_row = results_df.iloc[sample_idx]
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="stage-header">📸 Input Image</h2>', unsafe_allow_html=True)
        
        # Display image
        image_path = None
        if "image" in selected_row:
            image_path = selected_row["image"]
            if isinstance(image_path, str):
                img_path = Path(image_path)
                if not img_path.is_absolute():
                    # Try relative to radiology_report/files
                    radiology_root = project_root.parent / "radiology_report" / "files"
                    if radiology_root.exists():
                        # Extract relative path from full path
                        path_str = str(image_path)
                        if "files/p10" in path_str:
                            rel_path = path_str.split("files/p10/", 1)[-1]
                            img_path = radiology_root / "p10" / rel_path
                        else:
                            img_path = radiology_root / path_str
                
                if img_path.exists():
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=f"Image: {img_path.name}", use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")
                else:
                    st.error(f"Image not found: {img_path}")
                    st.info(f"Looking for: {image_path}")
            else:
                st.warning("No image path in results")
    
    with col2:
        st.markdown('<h2 class="stage-header">🤖 AI Predictions</h2>', unsafe_allow_html=True)
        
        # Impression
        if "impression" in selected_row:
            st.markdown("**📝 Impression:**")
            impression = selected_row["impression"]
            if pd.notna(impression) and str(impression).strip():
                st.markdown(f"<div class='result-box'>{impression}</div>", unsafe_allow_html=True)
            else:
                st.info("No impression available")
        
        # CheXpert predictions
        st.markdown("**🏷️ Predicted CheXpert Labels:**")
        pred_labels = {}
        for label in CHEXPERT13:
            # Check multiple column name formats
            pred_val = None
            for col_name in [label, f"{label}_pred", f"pred_{label}"]:
                if col_name in selected_row:
                    pred_val = selected_row[col_name]
                    break
            
            if pred_val is not None:
                pred_labels[label] = int(pred_val) if pd.notna(pred_val) else 0
            else:
                pred_labels[label] = 0
        
        # No Finding
        if "No Finding" in selected_row:
            pred_labels["No Finding"] = int(selected_row["No Finding"]) if pd.notna(selected_row["No Finding"]) else 0
        else:
            # Compute: 1 if all others are 0
            pred_labels["No Finding"] = 1 if sum(pred_labels.values()) == 0 else 0
        
        display_chexpert_labels(pred_labels, "Predictions")
    
    # Ground truth comparison
    st.markdown('<h2 class="stage-header">📋 Ground Truth Comparison</h2>', unsafe_allow_html=True)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("🏷️ Ground Truth Labels")
        gt_labels = {}
        for label in CHEXPERT14:
            # Check multiple formats
            gt_val = None
            for col_name in [label, f"{label}_gt", f"y_true_{label}"]:
                if col_name in selected_row:
                    gt_val = selected_row[col_name]
                    break
            
            if gt_val is not None:
                gt_labels[label] = int(gt_val) if pd.notna(gt_val) else 0
            else:
                gt_labels[label] = 0
        
        display_chexpert_labels(gt_labels, "Ground Truth")
    
    with col4:
        st.subheader("✅ Agreement Analysis")
        
        agreements = []
        disagreements = []
        
        for label in CHEXPERT13:
            pred_val = pred_labels.get(label, 0)
            gt_val = gt_labels.get(label, 0)
            
            if pred_val == gt_val:
                agreements.append(label)
            else:
                disagreements.append(f"{label}: GT={gt_val}, Pred={pred_val}")
        
        st.markdown(f"**Agreements:** {len(agreements)}/{len(CHEXPERT13)}")
        if agreements:
            st.markdown(", ".join(agreements[:5]))
            if len(agreements) > 5:
                st.markdown(f"... and {len(agreements) - 5} more")
        
        st.markdown(f"**Disagreements:** {len(disagreements)}/{len(CHEXPERT13)}")
        if disagreements:
            for item in disagreements[:5]:
                st.markdown(f"  - {item}")
            if len(disagreements) > 5:
                st.markdown(f"  ... and {len(disagreements) - 5} more")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: This demo uses CheXagent predictions from the 5k evaluation pipeline. Results are from the test set evaluation.")


if __name__ == "__main__":
    main()

