import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from deepface import DeepFace
import shap
import altair as alt
import plotly.express as px
from PIL import Image
import logging
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST

# Set up logging to capture errors internally
logging.basicConfig(filename='app_errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./models/distilbert_emotion')
model = DistilBertForSequenceClassification.from_pretrained('./models/distilbert_emotion')
label_encoder = joblib.load('./models/label_encoder.pkl')

# Wrap the model for SHAP compatibility
def model_predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.flatten().tolist()  # Convert 2D or 1D array to list
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# Initialize SHAP KernelExplainer
sample_texts = ['This is a sample text'] * 50  # Reduced for performance
background_data = np.array(sample_texts).reshape(-1, 1)  # Shape (50, 1) for SHAP
explainer = shap.KernelExplainer(model_predict, background_data)

# Emotion-emoji mapping
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Function to predict text emotion
def predict_emotions(docx):
    inputs = tokenizer(docx, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    # Validate pred_label
    if pred_label >= len(label_encoder.classes_):
        logging.error(f"Invalid pred_label: {pred_label}, max expected: {len(label_encoder.classes_) - 1}")
        pred_label = 0  # Fallback to first class
    return label_encoder.inverse_transform([pred_label])[0], probs[0].numpy(), pred_label

# Function to predict facial emotion
def predict_facial_emotion(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=True)
        return result[0]['dominant_emotion']
    except Exception as e:
        return f"Error in facial emotion detection: {str(e)}"

# Main Application
def main():
    st.title("Multimodal Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()

    if choice == "Home":
        add_page_visited_details("Home", datetime.now(IST))
        st.subheader("Emotion Detection in Text and Images")

        # Text input form
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Text Here")
            submit_text = st.form_submit_button(label='Submit Text')

        # Image upload
        uploaded_image = st.file_uploader("Upload an Image for Facial Emotion Detection", type=["jpg", "png"])

        if submit_text and raw_text:
            col1, col2 = st.columns(2)

            # Text prediction
            prediction, probability, pred_label = predict_emotions(raw_text)
            add_prediction_details(raw_text, prediction, float(np.max(probability)), datetime.now(IST))

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Text Emotion Prediction")
                st.write(f"Emotion: {prediction} {emotions_emoji_dict.get(prediction, '')}")
                st.write(f"Confidence: {np.max(probability):.2%}")

                # SHAP explanation with error handling
                st.success("Explainability (SHAP)")
                try:
                    input_data = np.array([raw_text]).reshape(1, 1)
                    shap_values = explainer.shap_values(input_data)
                    # Verify shap_values length
                    if len(shap_values) != len(label_encoder.classes_):
                        logging.error(f"Expected {len(label_encoder.classes_)} SHAP value arrays, got {len(shap_values)}")
                        raise ValueError("Incorrect number of SHAP value arrays")
                    # Create SHAP Explanation objects
                    tokenized_text = tokenizer(raw_text, padding=True, truncation=True, max_length=128, return_tensors='pt')
                    tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0], skip_special_tokens=True)
                    expected_value = model_predict(raw_text).mean(axis=0)  # Average prediction as base value
                    shap_explanation = [
                        shap.Explanation(
                            values=shap_vals,
                            base_values=expected_value[i],
                            data=tokens,
                            output_names=label_encoder.classes_[i]
                        ) for i, shap_vals in enumerate(shap_values)
                    ]
                    # Select the Explanation for the predicted emotion
                    shap_html = shap.plots.text(shap_explanation[pred_label], display=False)
                    st.components.v1.html(shap_html, height=600)
                except Exception as e:
                    logging.error(f"SHAP visualization failed: {str(e)}")
                    st.error("Unable to generate SHAP visualization. Please try again or contact support.")

            with col2:
                # Probability visualization
                st.success("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Emotion': label_encoder.classes_,
                    'Probability': probability
                })
                fig = px.bar(prob_df, x='Emotion', y='Probability', title='Emotion Probabilities')
                st.plotly_chart(fig)

        if uploaded_image:
            st.success("Facial Emotion Detection")
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            facial_emotion = predict_facial_emotion(np.array(image))
            st.write(f"Detected Facial Emotion: {facial_emotion}")

        # Ethical note
        st.markdown("""
        **Ethical Note**: This app uses AI to detect emotions, which may not always be accurate, especially for complex or culturally specific expressions. Potential biases in training data (e.g., Twitter or facial datasets) may affect results. Use predictions responsibly and avoid sensitive applications without further validation.
        """)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.subheader("Monitor")
        with st.expander("Page Visit Details"):
            details = view_all_page_visited_details()
            st.write(pd.DataFrame(details, columns=['Page Name', 'Time of Visit']))

        with st.expander("Prediction Details"):
            details = view_all_prediction_details()
            df = pd.DataFrame(details, columns=['Text', 'Prediction', 'Confidence', 'Time'])
            df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
            st.write(df)

    elif choice == "About":
        add_page_visited_details("About", datetime.now(IST))
        st.subheader("About")
        st.markdown("""
        This app detects emotions in text and images using advanced NLP (DistilBERT) and facial recognition (DeepFace). It supports eight emotions: anger, disgust, fear, joy, neutral, sadness, shame, and surprise. The app includes explainability features via SHAP to show which words influence predictions.
        """)

if __name__ == '__main__':
    main()