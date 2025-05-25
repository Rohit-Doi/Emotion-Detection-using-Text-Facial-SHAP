Emotion-Detection-using-Text-Facial-SHAP
A multimodal emotion detection application using DistilBERT for text emotion classification and DeepFace for facial emotion detection, with SHAP-based explainability for text predictions. Supports eight emotions: anger, disgust, fear, joy, neutral, sadness, shame, and surprise.
Description
This project combines natural language processing (NLP) and computer vision to detect emotions from text and facial images. It features:

Text Emotion Detection: Uses a fine-tuned DistilBERT model to classify emotions in text, with SHAP visualizations to explain predictions.
Facial Emotion Detection: Employs DeepFace to analyze emotions in uploaded images.
Monitoring: Tracks predictions and page visits using a SQLite database.
Ethical Considerations: Includes notes on potential biases and responsible use.

Setup

Clone the Repository:
git clone https://github.com/Rohit-Doi/Emotion-Detection-using-Text-Facial-SHAP.git
cd Emotion-Detection-using-Text-Facial-SHAP


Create and Activate a Virtual Environment:
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows


Install Dependencies:
pip install -r requirements.txt


Obtain Model and Data Files:

The models/ and data/ directories are excluded due to size. Download them from [insert Google Drive link or contact Rohit-Doi] or run train_distilbert.py to regenerate the model.
Place models/distilbert_emotion/ and models/label_encoder.pkl in the project root.


Run the Application:
streamlit run app.py



Usage

Text Emotion Detection: Enter text in the Streamlit UI to predict emotions and view SHAP visualizations.
Facial Emotion Detection: Upload a JPG/PNG image to detect emotions using DeepFace.
Monitor: View prediction history and page visits in the Monitor section.
About: Learn about the app’s functionality in the About section.

Project Structure

app.py: Main Streamlit application.
track_utils.py: Utilities for database tracking.
requirements.txt: List of dependencies.
tracker.db: SQLite database for tracking (auto-generated).
app_errors.log: Log file for errors (auto-generated).
models/ (excluded): Contains DistilBERT model and label encoder.
data/ (excluded): Contains training data.

Ethical Note
Emotion detection models may exhibit biases due to training data (e.g., Twitter or facial datasets). Results may not always be accurate, especially for complex or culturally specific expressions. Use predictions responsibly and avoid sensitive applications without validation.
Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
License
MIT License
Contact
For questions, contact [Rohit-Doi] via GitHub issues.
