
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Rock Paper Scissors AI",
    page_icon="✊",
    layout="wide"
)

# LOAD MODEL
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

# SIDEBAR
st.sidebar.title("Project Settings")

st.sidebar.info(
    "Deep Learning Image Classification"
)

st.sidebar.subheader("Training Results")

train_acc = st.sidebar.number_input(
    "Training Accuracy (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)

val_acc = st.sidebar.number_input(
    "Validation Accuracy (%)",
    min_value=0.0,
    max_value=100.0,
    value=0.0
)

train_loss = st.sidebar.number_input(
    "Training Loss",
    min_value=0.0,
    value=0.0
)

val_loss = st.sidebar.number_input(
    "Validation Loss",
    min_value=0.0,
    value=0.0
)

# HEADER
st.title("✊ Rock Paper Scissors AI")
st.subheader("Deep Learning Based Image Classification")

st.write(
    "An AI application using MobileNetV2 "
    "and Transfer Learning to recognize "
    "Rock, Paper and Scissors hand gestures."
)

st.divider()

# MODEL PERFORMANCE
st.header("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Training Accuracy", f"{train_acc:.2f}%")
c2.metric("Validation Accuracy", f"{val_acc:.2f}%")
c3.metric("Epochs", "8")

c4, c5 = st.columns(2)

c4.metric("Training Loss", f"{train_loss:.4f}")
c5.metric("Validation Loss", f"{val_loss:.4f}")

# MODEL ARCHITECTURE
st.divider()
st.header("🧠 Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.write("**Model:** MobileNetV2")
    st.write("**Pretrained Weights:** ImageNet")
    st.write("**Technique:** Transfer Learning")
    st.write("**Optimizer:** Adam")
    st.write("**Learning Rate:** 0.001")

with col2:
    st.write("**Epochs:** 8")
    st.write("**Batch Size:** 32")
    st.write("**Image Size:** 160 × 160")
    st.write("**Dropout:** 0.3")
    st.write("**Dataset Split:** 80% / 20%")

st.subheader("How Transfer Learning Works")

st.write("""
1. MobileNetV2 uses pretrained ImageNet weights.
2. The pretrained base layers are frozen.
3. A Global Average Pooling layer is added.
4. Dropout is used to reduce overfitting.
5. A Dense layer with 3 Softmax outputs classifies
   Rock, Paper and Scissors.
6. The new classification head is trained using Adam.
""")

# PARAMETERS
st.divider()
st.header("⚙️ Model Parameters")

st.write(
    f"Total Parameters: {model.count_params():,}"
)

st.write(
    f"Trainable Parameters: "
    f"{sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}"
)

st.write(
    f"Non-Trainable Parameters: "
    f"{sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights):,}"
)

# ACCURACY CHART
st.divider()
st.header("📈 Accuracy Comparison")

st.bar_chart({
    "Accuracy (%)": {
        "Training": train_acc,
        "Validation": val_acc
    }
})

# LOSS CHART
st.header("📉 Loss Comparison")

st.bar_chart({
    "Loss": {
        "Training": train_loss,
        "Validation": val_loss
    }
})

# PREDICTION
st.divider()
st.header("📸 Test Your Own Image")

uploaded_file = st.file_uploader(
    "Upload Rock, Paper or Scissors Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    if st.button("🔍 Predict Gesture"):

        img = image.resize((160, 160))

        img_array = np.array(img).astype("float32")

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Preprocessing is already inside model
        predictions = model.predict(
            img_array,
            verbose=0
        )

        index = np.argmax(predictions[0])

        predicted_class = class_names[index]

        confidence = float(
            predictions[0][index]
        ) * 100

        st.success(
            f"Prediction: {predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for i, name in enumerate(class_names):

            probability = float(predictions[0][i])

            st.write(
                f"{name.title()}: {probability * 100:.2f}%"
            )

            st.progress(probability)

# FOOTER
st.divider()

st.markdown("""
### Project Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning
- Streamlit

**Project:** Rock Paper Scissors Deep Learning

**Classification Classes:** Rock | Paper | Scissors
""")
