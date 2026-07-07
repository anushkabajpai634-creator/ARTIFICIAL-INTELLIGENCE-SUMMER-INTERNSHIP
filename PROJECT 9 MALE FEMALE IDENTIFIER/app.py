import streamlit as st
import cv
import joblib
import numpy as np
from PIL import Image

IMG_SIZE = 64

# Load trained model
model = joblib.load("PROJECT 9 MALE FEMALE IDENTIFIER/Male_Female_model.pkl")

st.title("Male vs Female Image Classification")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)

    # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.flatten().reshape(1, -1)

    prediction = model.predict(img)

    if prediction[0] == 0:
        st.success("Prediction: MALE")
    else:
        st.success("Prediction: FEMALE")
