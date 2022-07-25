# Sentiment-Prediction-App
A Streamlit app to predict statistics of sentiment scores of input.

Use command - " streamlit run app.py " to run the app.

In this we built a web app to predict emotions of the text which is given as input to the emotion classifier model (Logistic Regression) using a GUI based streamlit app. Streamlit is an open-source app framework for Machine Learning and Data Science. 

We built a pipeline consisting of CountVectorizer and LogisticRegression modules and pickled the trained model into a .pkl file and exported it to the web app. The app consists of 2 parts. One is to get the input from the user and the other is to display the results of the prediction. The text from the user is sent to the backend and given as input to the pipeline. 

The pipeline predicts the emotion and the probabilities of the emotions. The results, i.e. the emotion prediction for the given sentence, the probabilities of the emotion, confidence and a bar graph plotted against the sentiment and the probabilities are displayed in the dashboard using seaborn 
