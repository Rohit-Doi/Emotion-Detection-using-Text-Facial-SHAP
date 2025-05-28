# Emotion-Detection-using-Text-Facial-SHAP

<p align="center">
  <img src="images/homepage.png" alt="App Homepage" width="600"/>
  <br>
  <em>App Homepage Screenshot</em>
</p>

A multimodal emotion detection application using DistilBERT for text emotion classification and DeepFace for facial emotion detection, with SHAP-based explainability for text predictions. Supports eight emotions: anger, disgust, fear, joy, neutral, sadness, shame, and surprise.

## Description

This project combines natural language processing (NLP) and computer vision to detect emotions from text and facial images. It features:

- **Text Emotion Detection**: Uses a fine-tuned DistilBERT model to classify emotions in text, with SHAP visualizations to explain predictions.
- **Facial Emotion Detection**: Employs DeepFace to analyze emotions in uploaded images.
- **Monitoring**: Tracks predictions and page visits using a SQLite database.
- **Ethical Considerations**: Includes notes on potential biases and responsible use.

## Features

### Text Emotion Detection
<p align="center">
  <img src="images/text_emotion.png" alt="Text Emotion Detection" width="600"/>
  <br>
  <em>Text Emotion Detection Interface</em>
</p>

Enter text to predict emotions and view SHAP visualizations explaining the model's predictions.

### Facial Emotion Detection
<p align="center">
  <img src="images/facial_emotion.png" alt="Facial Emotion Detection" width="600"/>
  <br>
  <em>Facial Emotion Detection Interface</em>
</p>

Upload a JPG/PNG image to detect emotions using DeepFace.

### Monitoring
Track prediction history and page visits in the Monitor section.

### About
Learn about the app’s functionality and ethical considerations in the About section.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rohit-Doi/Emotion-Detection-using-Text-Facial-SHAP.git
   cd Emotion-Detection-using-Text-Facial-SHAP
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Obtain Model and Data Files**:
   - The `models/` and `data/` directories are excluded due to size constraints. Download them from [insert Google Drive link or contact Rohit-Doi] or run `train_distilbert.py` to regenerate the model.
   - Place `models/distilbert_emotion/` and `models/label_encoder.pkl` in the project root.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application.
- `track_utils.py`: Utilities for database tracking.
- `requirements.txt`: List of dependencies.
- `tracker.db`: SQLite database for tracking (auto-generated).
- `app_errors.log`: Log file for errors (auto-generated).
- `models/` (excluded): Contains DistilBERT model and label encoder.
- `data/` (excluded): Contains training data.
- `images/`: Screenshots for README documentation.

## Technologies Used

- **DistilBERT (Transformers)**: Text emotion classification.
- **DeepFace**: Facial emotion detection.
- **SHAP**: Explainability for text predictions.
- **Streamlit**: Interactive UI.
- **SQLite**: Prediction and visit tracking.
- **Python Libraries**: pandas, numpy, matplotlib, scikit-learn for data handling and visualization.

## Ethical Note

Emotion detection models may exhibit biases due to training data (e.g., Twitter or facial datasets). Results may not always be accurate, especially for complex or culturally specific expressions. Use predictions responsibly and avoid sensitive applications without validation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, contact [Rohit-Doi]