import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("flower_classification_model_vgg_2.h5")

# Class labels (update according to your training data)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Preprocessing function
def preprocess_image(img):
    img = img.resize((180, 180))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # RGBA to RGB
        img_array = img_array[:, :, :3]
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🌸 Flower Classification App")
st.write("Upload or capture a flower image. If it's not a flower, I'll let you know!")

option = st.radio("Choose image source:", ("Upload", "Camera"))

image = None

if option == "Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Camera":
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        image = Image.open(captured_image)

if image is not None:
    st.image(image, caption="Your Image", use_container_width=True)

    # Preprocess & Predict
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    max_prob = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    # Confidence check
    if max_prob < 0.9:
        st.markdown("### ❌ This doesn't look like a flower.")
    else:
        st.markdown(f"### 🌼 Predicted Flower: **{predicted_class.capitalize()}** ({max_prob:.2%} confidence)")
    
