
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Page configuration
st.set_page_config(
    page_title="Rock Paper Scissors AI",
    page_icon="✊",
    layout="centered"
)

# Custom heading
st.title("✊ Rock Paper Scissors AI")
st.write(
    "Upload your hand gesture image "
    "and let Deep Learning predict it!"
)

# Load trained model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(
        "rock_paper_scissors_model.keras",
        compile=False
    )

    with open("class_names.json", "r") as f:
        classes = json.load(f)

    return model, classes

model, class_names = load_my_model()

# Image upload
uploaded_file = st.file_uploader(
    "Upload Rock, Paper or Scissors Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Gesture"):

        with st.spinner("AI is predicting..."):

            # Resize image
            img = image.resize((160, 160))

            # Convert to array
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Preprocessing
            img_array = preprocess_input(
                img_array.astype("float32")
            )

            # Prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                predictions[0]
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = float(
                predictions[0][predicted_index]
            ) * 100

        st.success(
            f"Prediction: {predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for i, name in enumerate(class_names):

            st.write(
                f"{name.title()}: "
                f"{predictions[0][i] * 100:.2f}%"
            )

            st.progress(
                float(predictions[0][i])
            )

st.markdown("---")

st.caption(
    "Deep Learning Project | MobileNetV2 | TensorFlow"
)
