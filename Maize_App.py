import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# 🎨 Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #2e7d32;
        margin-top: -20px;
    }

    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #555;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        margin-top: 20px;
    }

    .upload-section {
        padding: 20px;
        background: white;
        border-radius: 10px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model("maize_FINAL.keras")
    class_names = [
        'Maize fall armyworm',
        'Maize grasshopper',
        'Maize healthy',
        'Maize leaf beetle',
        'Maize leaf blight',
        'Maize leaf spot',
        'Maize streak virus'
    ]
    return model, class_names


model, class_names = load_model()

# -------------------------------------------
# Sidebar
# -------------------------------------------
st.sidebar.title(" How to Use")
st.sidebar.info("""
1. Upload a maize leaf image.  
2. Wait for the model to analyze it.  
3. View the predicted disease & confidence score.  

✔ Recommended: Use clear leaf images  
""")

st.sidebar.markdown("---")
st.sidebar.write("👨 **Maize Disease Detection Model**")

# -------------------------------------------
# Main Title
# -------------------------------------------
st.markdown("<h1 class='main-title'> Maize Leaf Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Diagnosis for common maize leaf conditions.</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------
# File Upload Section
# -------------------------------------------
with st.container():
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" Upload Maize Leaf Image", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------
# Processing
# -------------------------------------------
if uploaded_file is not None:
    st.subheader(" Preview")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Convert uploaded file
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array)
    pred_class_index = np.argmax(pred)
    pred_class = class_names[pred_class_index]
    confidence = np.max(pred)

    # Display results
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown("###  Prediction Results")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence Score:** {confidence:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    # Optional: confidence progress bar
    st.write("###Confidence Level")
    st.progress(float(confidence))
