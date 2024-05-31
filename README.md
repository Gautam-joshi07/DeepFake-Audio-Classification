# DeepFake-Audio-Classification


This project classifies audio files as either "real" or "fake" using a deep learning model. The project pipeline includes data ingestion, base model, model training, model evaluation, and deployment using Flask. The following tools and technologies are used:


* MLflow: For parameter tracking and model management.
* DVC (Data Version Control): For pipeline tracking.
* Git: For source code management.




## Description

- This project is developed to solve the problem of classifying a audio file in real or fake.

- Implemented the project using deep learning



## Project Structure


* main.py: The entry point for running the entire pipeline, from data ingestion to model evaluation.
* app.py: The Flask application for uploading audio files and classifying them as "real" or "fake".
* requirements.txt: A list of Python packages required to run the project.
* artifacts/: Directory containing the trained model, base model, Dataset.
* audio_files/: Directory where uploaded audio files are stored.
* static/: Directory where generated spectrogram images are stored.
* templates/: Directory containing HTML templates for the Flask app.
* mlruns/: Directory that tracks all the parameter and models.
* logs/: Directory that contain the logs.
## Setup

Prerequisites
* Python 3.8 or higher
* pip package installer
* virtualenv (optional but recommended for creating a virtual environment)
* git for source code management
* dvc for data version control
* mlflow for experiment tracking

## Installation

### Clone the Repository:

```bash
 git clone https://github.com/yourusername/deepfake-audio-classification.git
 cd deepfake-audio-classification
```
### Create a Virtual Environment:
```bash
python3 -m venv venv
source venv/bin/activate 

```

### Requirements
```bash 
pip install -r requirements.txt
```


## Running the pipeline

### Run the Main Pipeline:

* The main.py script handles the entire deep learning pipeline, from data ingestion to model evaluation. Simply run:

```bash
python main.py
```
This will execute the following steps:

* Data ingestion
* Data preprocessing
* Model training
* Model evaluation

### Run the Flask Application:

### The app.py script starts a Flask web server for uploading and classifying audio files:
```bash
python app.py
```


### The streamlit_app.py starts a streamlit app:

```bash 

streamlit run streamlit_run.app
```


## You can Try on below Link !

https://huggingface.co/spaces/Joshi07/DeepFake_audio_Classification



## MLflow

* Use MLflow for Parameter Tracking and Model Management
* MLflow is used for tracking experiments and managing models. To start the MLflow server, run:

```bash
mlflow ui
```
* By default, the MLflow UI will be accessible at http://127.0.0.1:5000/

## DVC

* DVC is used to version control the data and manage the pipeline stages. To set up DVC, follow these steps:

* Initialize DVC:
```bash
dvc init
```


## Feedback
If you have any feedback, please reach out to us at gautampjoshi7@gmail.com


