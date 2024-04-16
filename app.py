import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import InputLayer

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    try:
        # Define a custom object dictionary to handle custom layers
        custom_objects = {'InputLayer': InputLayer}
        model = load_model('potatoes.h5', custom_objects=custom_objects)
        return model, None
    except Exception as e:
        return None, str(e)

model, load_error = load_trained_model()

if model is None:
    st.error(f"Failed to load the model. Error: {load_error}")
    st.stop()

# Function to preprocess the image
def preprocess_image(image_file):
    try:
        img = image.load_img(image_file, target_size=(175, 175))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

# Function to make predictions
def predict_disease(image):
    processed_image, preprocess_error = preprocess_image(image)
    if processed_image is not None:
        prediction = model.predict(processed_image)
        return prediction, None
    else:
        return None, preprocess_error

# Streamlit app
def main():
    st.title("Potato Diseases Classification")
    st.write("Upload an image of a potato leaf to classify its disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        if st.button('Classify'):
            st.write("Classifying...")
            try:
                prediction, predict_error = predict_disease(uploaded_file)
                if prediction is not None:
                    classes = ['Early Blight', 'Late Blight', 'Healthy']
                    predicted_class = classes[np.argmax(prediction)]
                    st.write(f"Prediction: {predicted_class}")
                else:
                    st.error(f"Failed to classify the image. Error: {predict_error}")
            except Exception as e:
                st.error("An error occurred during prediction.")
                st.error(str(e))

if __name__ == '__main__':
    main()
