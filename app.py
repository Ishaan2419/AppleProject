import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model if not present
if not os.path.exists("fruit_model.keras"):
    url = "https://drive.google.com/uc?id=106hP_fAg-_sez85rh1NHZ8iTmwDafbYO"
    gdown.download(url, "fruit_model.keras", quiet=False)

# Load trained model
model = tf.keras.models.load_model("fruit_model.keras", compile=False, safe_mode=False)

class_names = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges"
]

st.title("Fruit Freshness Classifier")

uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write("Prediction:", predicted_class)
    st.write("Confidence:", confidence)
