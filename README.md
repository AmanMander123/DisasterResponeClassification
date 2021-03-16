# Disaster Response ETL, Machine Learning Pipeline + Flask Web App

## Motivation
The purpose of this app is to classify disaster related messages. A machine learning model is used to aid an emergency worker in classifying an input message.

## Code / Data
- data/process_data.py: Extracts data from messages.csv (message data) and categories.csv (message classes) and creates a SQLite database containing the cleaned and merged data.
- models/train_classifier.py: Takes the SQLite database as the input and uses the data to train and tune a machine learning model in order to categorize the messages. The file outputs a pickle file containing the fitted and tuned model. After the training process, the evaluation metrics are printed.
- app/run.py: Flask app that runs the web app.
- templates/: Folder containing the necessary html files for the web app front end.

## Run
- Install the requirements using the requirements.txt file
- Navigate to the data directory and run: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
- Navigate to the models directory and run: python train_classifier.py ../data/DisasterResponse.db classifier.pkl
- Navigate to the app directory and run: python run.py
- Follow the instructions on the terminal in order to run the web app

