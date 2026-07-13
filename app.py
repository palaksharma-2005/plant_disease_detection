import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Load model only once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "plant_disease_model_new.h5",
        compile=False
    )
    return model

model = load_model()

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# App title
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess image
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Display result
    st.success(f"🌱 Predicted Disease: **{class_names[predicted_index]}**")
    st.info(f"🎯 Confidence: **{confidence:.2f}%**")
