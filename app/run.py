import json
import plotly
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify

import numpy as np
import operator
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import re
from collections import Counter

# initializing Flask app
app = Flask(__name__)

def tokenize(text):
    """
    Inputs:
    text: string messages
    Outputs:
    list: tokenized words
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)

    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    words = [lemmatizer.lemmatize(word, pos='v') for word in words]

    return words

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df[df.columns[4:]].head()

    cat_p = df[df.columns[4:]].sum()/len(df)
    cat_p = cat_p.sort_values(ascending = False)

    cats = list(cat_p.index)

    words_with_repetition=[]

    for text in df['message'].values:
        tokenized_ = tokenize(text)
        words_with_repetition.extend(tokenized_)

    word_count_dict = Counter(words_with_repetition)

    sorted_word_count_dict = dict(sorted(word_count_dict.items(),
                                         key=operator.itemgetter(1),
                                         reverse=True))

    top, top_20 =0, {}

    for k,v in sorted_word_count_dict.items():
        top_20[k]=v
        top+=1
        if top==20:
            break
    words=list(top_20.keys())

    count_props=100*np.array(list(top_20.values()))/df.shape[0]
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cats,
                    y=cat_p
                )
            ],
            'layout': {
                'title': 'Frequency of Messages by Category',
                'yaxis': {
                    'title': "Frequency",
                    'automargin':True
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -40,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=count_props
                )
            ],

            'layout': {
                'title': 'Frequency of top 20 words in percentage',
                'yaxis': {
                    'title': 'Frequency (Out of 100)',
                    'automargin': True
                },
                'xaxis': {
                    'title': 'Top 20 words',
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphsJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphsJSON=graphsJSON, data_set=df)

# web page that handles user query and displays model results
@app.route('/go')

def go():

    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html',
                            query=query,
                            classification_result=classification_results
                          )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
