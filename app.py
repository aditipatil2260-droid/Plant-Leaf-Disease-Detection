import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Plant Leaf Disease Detection", page_icon="🌿")

st.title("🌿 Plant Leaf Disease Detection")
st.write("Upload a plant leaf image to predict its disease.")

model = load_model("plant_leaf_disease_model.keras")

with open("class_names.json", "r") as f:
    classes = json.load(f)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_container_width=True)

    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Disease"):
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions)]
        confidence = float(np.max(predictions)) * 100

        st.success(f"Predicted Disease: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}%")