import streamlit as st
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os
from pathlib import Path
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="🌪️ Disaster Classification AI",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* (Your CSS remains the same) */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .disaster-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
    }
    .confidence-high { color: #00ff84; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# --- Constants and File Paths ---
DISASTER_ICONS = {
    'Water_Disaster': '🌊',
    'Land_Slide': '🏔️',
    'Urban_Fire': '🔥',
    'Earthquake': '🌋'
}
DISASTER_COLORS = {
    'Water_Disaster': '#3b82f6',
    'Land_Slide': '#8b5cf6',
    'Urban_Fire': '#ef4444',
    'Earthquake': '#f59e0b'
}
CLASS_NAMES = ['Earthquake', 'Land_Slide', 'Urban_Fire', 'Water_Disaster']

# Path to the root of the project (one level up from this script)
PROJECT_ROOT = Path(__file__).parent.parent

MODEL_BEFORE_PATH = PROJECT_ROOT / "backend" / "models" / "effnet_before.keras"
MODEL_TUNED_PATH = PROJECT_ROOT / "backend" / "models" / "effnet_tune.keras"
SAMPLE_IMG_DIR = PROJECT_ROOT / "frontend" / "samples"


# --- Model Loading and Preprocessing ---
@st.cache_resource
def load_models():
    """Load the disaster classification models using corrected paths"""
    try:
        if not MODEL_BEFORE_PATH.exists() or not MODEL_TUNED_PATH.exists():
            st.error("Model files not found. Please check the paths.")
            return None, None
        
        with st.spinner("Loading AI models... This may take a moment."):
            model_before = load_model(str(MODEL_BEFORE_PATH))
            model_tuned = load_model(str(MODEL_TUNED_PATH))
        
        st.success("✅ Models loaded successfully!")
        return model_before, model_tuned
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    image = image.resize((260, 260)).convert('RGB')
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

def get_prediction(model, img_array: np.ndarray) -> dict:
    """Get prediction from a loaded model"""
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_class_idx = np.argmax(predictions)
    return {
        "predicted_class": CLASS_NAMES[predicted_class_idx],
        "confidence": float(predictions[predicted_class_idx]),
        "all_predictions": {CLASS_NAMES[i]: float(p) for i, p in enumerate(predictions)}
    }

# --- UI Components ---
def create_prediction_chart(predictions_dict, title):
    """Create a bar chart for predictions"""
    labels = list(predictions_dict.keys())
    probs = [p * 100 for p in predictions_dict.values()]
    colors = [DISASTER_COLORS.get(cls, '#808080') for cls in labels]
    
    fig = go.Figure(go.Bar(
        x=probs, y=labels, orientation='h',
        marker_color=colors, text=[f"{p:.1f}%" for p in probs]
    ))
    fig.update_layout(
        title=title, xaxis_title="Confidence (%)",
        height=300, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def get_confidence_color_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.8: return "confidence-high"
    if confidence >= 0.6: return "confidence-medium"
    return "confidence-low"

def display_sample_images():
    """Display sample images for user to select"""
    st.markdown("#### 📂 Or Try a Sample Image")
    if not SAMPLE_IMG_DIR.is_dir():
        st.warning("Sample images directory not found.")
        return None

    sample_files = list(SAMPLE_IMG_DIR.glob("*.jpg"))
    if not sample_files:
        st.warning("No sample images found.")
        return None

    # Initialize session state for selected sample
    if 'selected_sample_path' not in st.session_state:
        st.session_state.selected_sample_path = None

    cols = st.columns(len(sample_files))
    for i, file in enumerate(sample_files):
        with cols[i]:
            image = Image.open(file)

            disaster_name_from_file = file.stem
            display_name = disaster_name_from_file.replace('_', ' ')
            icon = DISASTER_ICONS.get(disaster_name_from_file, '🌪️')

            st.info(f"{icon} {display_name}")
            st.image(image, use_container_width=True)
            if st.button(f"Use Sample {i+1}", key=f"sample_{i}", use_container_width=True):
                st.session_state.selected_sample_path = str(file)
                st.rerun()
    
    # Return the selected sample image if one was chosen
    if st.session_state.selected_sample_path:
        return Image.open(st.session_state.selected_sample_path)
    return None

def display_results(result, model_type_str):
    """Display the prediction results in a formatted card and chart."""
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    icon = DISASTER_ICONS.get(predicted_class, '🌪️')
    confidence_class = get_confidence_color_class(confidence)

    st.markdown(f"""
    <div class="main-header">
        <h4>{model_type_str} Model</h4>
    </div>          
    <div class="result-card">
        <h4>{icon} {predicted_class.replace('_', ' ')}</h4>
        <p class="{confidence_class}">Confidence: {confidence * 100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    fig = create_prediction_chart(
        result['all_predictions'],
        f"{model_type_str} Model Predictions"
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Main Application Logic ---
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌪️ Disaster Classification AI</h1>
        <p>Upload an image or choose a sample to classify the disaster type using a fine-tuned EfficientNet model.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔧 Options")
        use_api = st.radio(
            "Prediction Method:", ("Direct Model Integration", "FastAPI Server"),
            help="Choose how to perform predictions. 'Direct' loads models into Streamlit, 'FastAPI' calls a separate server."
        )
        api_url = "http://127.0.0.1:8000"
        if use_api == "FastAPI Server":
            api_url = st.text_input("FastAPI Server URL", value=api_url)
            if st.button("Test API Connection"):
                with st.spinner("Pinging API..."):
                    try:
                        response = requests.get(f"{api_url}/health")
                        if response.status_code == 200:
                            st.success("✅ API server is running!")
                            st.json(response.json())
                        else:
                            st.error(f"❌ API server returned status {response.status_code}.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Connection failed: {e}")
        
        st.markdown("---")
        st.info("This project utilizes FastAPI as the backend to retrieve predictions from the model, while the streamlit is used to serve the frontend.")


    # --- Image Selection ---
    st.header("1. Select an Image")
    uploaded_file = st.file_uploader(
        "Upload your own image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to classify."
    )
    
    # Clear selected sample when a new file is uploaded
    if uploaded_file and 'selected_sample_path' in st.session_state:
        st.session_state.selected_sample_path = None
    
    sample_image = display_sample_images()
    
    image_to_use = None
    image_source = ""
    
    if uploaded_file:
        image_to_use = Image.open(uploaded_file)
        image_source = "Uploaded Image"
    elif sample_image:
        image_to_use = sample_image
        image_source = "Sample Image"

    if image_to_use:
        st.header("2. Get Predictions")
        col1, col2 = st.columns([0.6, 1.4])
        with col1:
            st.image(image_to_use, caption=image_source, use_container_width=True)

        with col2:
            tab1, tab2 = st.tabs(["⚖️ Model Comparison", "🔍 Single Prediction"])

            # --- Comparison Tab ---
            with tab1:
                if st.button("⚖️ Compare Both Models", type="primary", use_container_width=True):
                    if use_api == "FastAPI Server":
                        img_bytes = io.BytesIO()
                        image_to_use.save(img_bytes, format='PNG')
                        files = {'file': ('image.png', img_bytes.getvalue(), 'image/png')}
                        with st.spinner("Comparing models via API..."):
                            try:
                                response = requests.post(f"{api_url}/compare", files=files)
                                if response.ok:
                                    results = response.json()
                                    display_results(results['before_tuning'], "Base")
                                    display_results(results['after_tuning'], "Fine-Tuned")
                                else:
                                    st.error(f"API Error: {response.text}")
                            except Exception as e:
                                st.error(f"Failed to call API: {e}")
                    else: # Direct Integration
                        model_before, model_tuned = load_models()
                        if model_before and model_tuned:
                            with st.spinner("Comparing models directly..."):
                                img_array = preprocess_image(image_to_use)
                                result_before = get_prediction(model_before, img_array)
                                result_tuned = get_prediction(model_tuned, img_array)
                                display_results(result_before, "Base")
                                display_results(result_tuned, "Fine-Tuned")

            # --- Single Prediction Tab ---
            with tab2:
                model_type = st.selectbox(
                    "Select Model", ("tuned", "before"),
                    format_func=lambda x: "Fine-Tuned Model" if x == "tuned" else "Base Model"
                )
                if st.button("🔍 Predict with Selected Model", use_container_width=True):
                    if use_api == "FastAPI Server":
                        img_bytes = io.BytesIO()
                        image_to_use.save(img_bytes, format='PNG')
                        files = {'file': ('image.png', img_bytes.getvalue(), 'image/png')}
                        data = {'model_type': model_type}
                        with st.spinner("Predicting via API..."):
                            try:
                                response = requests.post(f"{api_url}/predict", files=files, data=data)
                                if response.ok:
                                    display_results(response.json(), "Fine-Tuned" if model_type == "tuned" else "Base")
                                else:
                                    st.error(f"API Error: {response.text}")
                            except Exception as e:
                                st.error(f"Failed to call API: {e}")
                    else: # Direct Integration
                        model_before, model_tuned = load_models()
                        if model_before and model_tuned:
                            selected_model = model_tuned if model_type == "tuned" else model_before
                            with st.spinner("Predicting directly..."):
                                img_array = preprocess_image(image_to_use)  # Fixed typo here
                                result = get_prediction(selected_model, img_array)
                                display_results(result, "Fine-Tuned" if model_type == "tuned" else "Base")

    # Add a button to clear selected sample
    if 'selected_sample_path' in st.session_state and st.session_state.selected_sample_path:
        if st.button("🗑️ Clear Selected Sample"):
            st.session_state.selected_sample_path = None
            st.rerun()

    # Bottom section
    st.markdown("---")
    st.header("Model Information")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("Model Details")
        st.write("""
        This streamlit app is a demonstration of an EfficientNetB2 model (a pretrained model) used to classify images for Disaster Classification.
        
        This model is trained with:
        - Oversampling Method
        - Data Augmentation
        - Fine-Tuning Model
        
        Oversampling was done only to the training dataset. Augmentation was done to training data generator, while testing and validation was only scaled.
        """)

    with col_info2:
        st.subheader("Training Process")
        st.write("""
        Training Process includes:
        1. Oversampling to balance classes
        2. Data Augmentation with ImageDataGenerator (Rotation, Zoom, Horizontal Flip, etc)
        3. Training with Adam Optimizer
        4. Fine-Tuning by lowering learning rate and Unfreezing layers
        
        EfficientNetB2 model was trained with a total of 1500 images across 4 classes and achieved a training and testing accuracy of 96.99% and 95.38%.
        """)
        
    st.markdown("---")
    st.caption("© 2025 Disaster Image Classification | Created with Streamlit |  Author: Richard Dean Tanjaya ")


if __name__ == "__main__":
    main()