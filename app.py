import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    try:
        model = load_model('potatoes.h5')
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
        img = Image.open(image_file)
        img = img.resize((175, 175))  # Resize image to match input size of the model
        img_array = np.array(img) / 255.0   # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
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
