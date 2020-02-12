# Disaster Response Pipeline

## 1. Overview

In this project, I applied data engineering and machine learning pipelines to analyze disaster data from Figure Eight and built a model for an API that classifies disaster messages.

The **data** directory contains a data set which are real messages that were sent during disaster events. I created a machine learning pipeline to categorize these events so that related agency could be assigned for support. This project also includes a web app where an emergency worker can input a message and get classification results in categories. The web app also displays visualizations of the data.

## 2. Project Components

There are three components of this project:

### 2.1. ETL Pipeline

File **data/process_data.py** contains data cleaning pipeline that:

- Loads and merge the `messages` and `categories` dataset
- Perform data cleaning and store the cleaned table in a **SQLite database**

### 2.2. ML Pipeline

File **models/train_classifier.py** contains the machine learning pipeline that:

- Loads and splits data from the **SQLite database**
- Builds a machine learning pipeline
- Trains models and select the best one using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

### 2.3. Flask Web App

Running [this command](#com) from the app directory will start the web app where users can enter their query message, and then app will classify the text message into categories so that appropriate agencies can be reached out for help.

## 3. Running

There are three steps to get up and runnning with the web app if you want to start from ETL process.

### 3.1. Data Cleaning

**Set your current location to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _data_ folder but the above command will still run and replace the file with same information.

### 3.2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

Please this is a slow process. This piece of script used the cleaned data to train the model, selected the model with the best predictive power using grid search and saved the model to a pickle file.

### 3.3. Starting the web app

Please **Locate to the app directory** and run the following command:

```bat
python run.py
```

## 4. Conclusion

Please see the training data set as seen on the main page of the web app.

## 5. Requirements

This project should be able to run in **Python 3**, and please download the necessary packages.

## 6. Credits and Acknowledgements

This is project completed based on the Udacity Data Scientist Nanodegree program.
