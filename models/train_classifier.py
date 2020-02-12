
# imports
import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import re
import sys
import nltk
# reminder to download everytime changing the file location
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine
import time
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Inputs:
    database file path

    Returns:
    X: dataframe training set
    Y: dataframe testing set
    names list: column names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', con = engine)
    X,Y = df['message'], df.iloc[:,4:]

    # mapping three distinct values to 1
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    names = Y.columns

    return X, Y, names

def tokenize(text):
    """
    Inputs:
    text: string Messages

    Returns:
    words: normalized, tokenized and lemmatized words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = word_tokenize(text)

    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    # lemmatize words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """
    build models
    """
    # define pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    # define hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create models
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Inputs:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file
    Inputs:
    model: Trained model
    model_filepath: Filepath to save the model
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Execute all functions above
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model successfully saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
