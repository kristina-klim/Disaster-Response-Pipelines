import sys
import os
import re
import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle

from sqlalchemy import create_engine
def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath of file within sql database data/DisasterResponseDb.db
    OUTPUT:
    X - array of message data (input data for model)
    Y - array of categorisation of messages (target variables)
    category_names - names of target variables
    Loads data from sql database and extracts information for modelling
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    
    df = pd.read_sql_table('disaster_response', engine)


    X = df['message']
    y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, y, category_names


def tokenize(text):
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    clean_tokens = []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w  not in stopwords.words("english") ]
    lemmatizer = WordNetLemmatizer()
    for word in words:
        lemmatized_word = lemmatizer.lemmatize(word)
        clean_tokens.append(lemmatized_word)
    return clean_tokens


def build_model():
    '''
    Function to build model pipeline with feature extraction and estimator.
    
    ARGS:
    None
    
    OUTPUT:
    pipeline: built model
    
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    parameters = {'clf__estimator__criterion': ["gini", "entropy"],     
        'clf__estimator__n_jobs':[-1],
        'clf__estimator__n_estimators': [5]
        }
    cv = GridSearchCV(estimator=pipeline, 
                      param_grid=parameters, 
                      cv=2, verbose=3
                      )
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    Function to print out evaluation of trained model on test data.
    
    ARGS:
    model: trained model
    X_test: messages test data
    Y_test: output categories test data
    category_names: name of output categories
    
    OUTPUT:
    None
    
    '''
    
    y_pred = model.predict(X_test)
    
    index = 0
    for category in category_names:
        print("output category in column {}: {}".format(index, category))
        evaluation_report = classification_report(y_test[:,index], y_pred[:,index])
        index += 1
        print(evaluation_report)



def save_model(model, model_filepath):
    '''
    Function to export model as a pickle file.
    
    ARGS:
    model: trained model
    model_filepath: path and filename to save pickle file of trained model
    
    OUTPUT:
    None
    
    '''
    filename = 'classifier.pkl'
    pickle.dump(model, open(filename, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    