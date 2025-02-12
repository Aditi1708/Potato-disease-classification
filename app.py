import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
import os

# Function to download the model file if not present
def download_model(url, output_path):
    if not os.path.exists(output_path):
        with st.spinner('Downloading model... Please wait...'):
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success('Model downloaded successfully!')
            else:
                st.error(f'Failed to download the model file. Status code: {r.status_code}. Check the URL or network.')
                return False
    return True

# Set the path and URL for the model
model_path = 'potatoes.h5'
model_url = 'https://drive.google.com/uc?export=download&id=1OvyHsGPReRkt-krG207BvgXynjiXGR9Y'

# Ensure model is downloaded
if download_model(model_url, model_path):
    # Load the trained model
    try:
        model = load_model(model_path)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.stop()

# Function to preprocess the image
def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_disease(image_file):
    processed_image = preprocess_image(image_file)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        if st.button('Classify'):
            with st.spinner('Classifying...'):
                prediction = predict_disease(uploaded_file)
                disease_class = np.argmax(prediction)
                if disease_class == 0:
                    st.write("Prediction: Early Blight")
                elif disease_class == 1:
                    st.write("Prediction: Late Blight")
                else:
                    st.write("Prediction: Healthy Potato Leaf")

if __name__ == '__main__':
    main()
