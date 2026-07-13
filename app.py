
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "plant_disease_model_new.h5",
        compile=False
    )
    return model

model = load_model()
)

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to detect its disease.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Predicted Disease: {class_names[predicted_index]}")
    st.info(f"Confidence: {confidence:.2f}%")
