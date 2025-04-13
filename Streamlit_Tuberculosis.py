import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="TB Detection", layout="centered")

# Load your trained VGG16 model
@st.cache_resource
def load_vgg16_model():
    return load_model("C:/Users/Admin/OneDrive/Documents/Tuberculosis project/VGG16_tb_detector.h5")  # Your saved model

model = load_vgg16_model()


# st.title("Tuberculosis Detection using VGG16")
# st.write("Upload a chest X-ray image to check for TB.")


st.sidebar.title("Navigate through the sections")
choice = st.sidebar.radio(label = "Select a section", options=["Home", "Tuberculosis detection"])
                                                               
if choice == 'Home' :


    st.markdown("## 🩺 Tuberculosis Detection Using VGG16")
    st.markdown("""
                **Tuberculosis (TB)** is a potentially serious infectious disease that primarily affects the lungs. It is caused by the bacterium *Mycobacterium tuberculosis* and remains one of the top 10 causes of death worldwide. Early and accurate detection of TB is crucial for effective treatment and preventing its spread.

                Chest X-rays are commonly used in screening, but interpretation requires trained professionals. This application uses **deep learning** to assist in detecting TB from X-ray images.

                ### 🤖 Why VGG16?
                - **Transfer Learning**: Pre-trained on ImageNet, VGG16 extracts strong features even from small medical datasets.
                - **Deep Feature Extraction**: Captures fine-grained patterns and structural details.
                - **Good Generalization**: Performs well on unseen data after fine-tuning.
                - **Deployable**: Easy to integrate into apps and services like this one.
                """)

elif choice == 'Tuberculosis detection':

    # File uploader
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Predict TB Status"):

            # Preprocess the image
            img = np.array(image_pil)
            img = cv2.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # Normalize the image
            
            # Make prediction
            prediction = model.predict(img)
            result = "Tuberculosis Detected" if prediction[0][0] > 0.5 else "Normal"
            confidence = prediction[0][0] if result == "Tuberculosis Detected" else 1 - prediction[0][0]
            
            # Display result
            st.write(f"### Prediction: {result}")
            st.write(f"Confidence: {confidence:.2%}")


    