# Project: Disaster Response Pipeline Project

This project is a part of the Udacity Nanodegree educational program.

## Project Intro/Objective

Over the past few years, many cataclysms have occurred on Earth: fires, earthquakes, pandemics, heat, and much more. It is more important than ever to respond quickly to messages from the victims.
This project-based on data comes from Appen-FigureEight Inc and uses real messages from people who need help. 
The basis of this project is a web - app that determines which of the 36 categories the message belongs to.

## Methods Used

•	Pandas, Numpy, Scikit-Learn for cleaning, manipulation, and transform data
•	NLTK for text processing 
•	Plotly for data visualization
•	Flask for web-app  development


## Needs of this project:

•	data exploration statistics
•	data cleaning
•	data visualization
•	nlp processing
•	pipeline building
•	machine-learning algorithm

## Getting Started

The files used in this project:
-	__app__
-	| master.html  _# main page of web app_
-	| go.html  _# classification result page of web app_
-	| run.py  _# web-app back-end information based on Flask_

-	__data__
-	| disaster_categories.csv 
-	| disaster_messages.csv  
-	| process_data.py algorithm that clean, transform and save text file as a SQL table
-	| InsertDatabaseName.db   _# database to save clean data to_

-	__models__
-	| train_classifier.py takes the SQL table, implements ML models and save as pickle file
-	| classifier.pkl  _# saved model_

-	README.md


## How to run:
•	python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
•	python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
•	cd /home/workspace/app
•	python run.py
•	env|grep WORK
•	Go to http://0.0.0.0:3001/ to see the results of the project


## Screenshot of web app
# <img width="678" alt="ScreenShot- Disaster-Response-Pipelines" src="https://user-images.githubusercontent.com/84743536/131203065-6a69ec04-0f56-47d8-a47c-a16ca7a0da80.png">


