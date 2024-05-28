# DeepFake-Audio-Classification


Deepfake Audio Classification
This project classifies audio files as either "real" or "fake" using a deep learning model. The project pipeline includes data ingestion, base model, model training, model evaluation, and deployment using Flask. The following tools and technologies are used:

MLflow: For parameter tracking and model management.
DVC (Data Version Control): For pipeline tracking.
Git: For source code management.


Project Structure

main.py: The entry point for running the entire pipeline, from data ingestion to model evaluation.
app.py: The Flask application for uploading audio files and classifying them as "real" or "fake".
requirements.txt: A list of Python packages required to run the project.
artifacts/: Directory containing the trained model, base model, Dataset.
audio_files/: Directory where uploaded audio files are stored.
static/: Directory where generated spectrogram images are stored.
templates/: Directory containing HTML templates for the Flask app.
mlruns/: Directory that tracks all the parameter and models.
logs/: Directory that contain the logs.

Setup
Prerequisites
Python 3.7 or higher
pip package installer
virtualenv (optional but recommended for creating a virtual environment)
git for source code management
dvc for data version control
mlflow for experiment tracking
