import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model if not present
if not os.path.exists("fruit_model.h5"):
    url = "https://drive.google.com/uc?id=1FtlP_6YBTxPst-_gWEhlVta4dot80H65"
    gdown.download(url, "fruit_model.h5", quiet=False)

# Load model
model = tf.keras.models.load_model("fruit_model.h5", compile=False)

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
