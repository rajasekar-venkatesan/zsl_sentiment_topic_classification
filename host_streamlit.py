# Imports
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from feasibility_study import model


# Main
"""
# Feasibility Study
## Models
The following 2 models can be tested using this demo.
### 1. Sentiment Analysis
Given a text, try to find out whether the text entered is Positive, Neutral or Negative
### 2. Topic Classification
Given a text, try to find out whether the text entered belongs to Financial Counselling, Clinical Competency, Waiting Time, Medical Mismanagement or Others
"""


raw_text = st.text_area("Enter text: ", "I was hospitalized last week. The hospital was well maintained but too crowded. Their service was good. I had to wait for long time before seeing the doctor.")
sentences = [item.strip() for item in raw_text.split(".") if len(item.strip())>1]
choice = st.selectbox("Choose Analysis Level: ", ['Document Level', 'Sentence Level'])
if choice == 'Sentence Level':
    sentences = {i: sent.text for i, sent in enumerate(sentences)}
    st.write(sentences)
    sentence_choice = st.selectbox("Choose a sentence to analyze: ", list(sentences.keys()))
    text = sentences.get(int(sentence_choice))
else:
    text = raw_text

"""
## 1. Sentiment Analysis
"""
results, prediction = model.get_sentiment(text)
sentiment_scores = {k: v for k, v in zip(results['labels'], results['scores'])}
x = model.sentiment_labels
x_pos = np.arange(len(x))
y = [sentiment_scores.get(item) for item in x]

f"""
### Sentiment Verdict: {prediction}

Model Internals
"""
fig = plt.figure(figsize=(10, 5))
plt.bar(x_pos, y, color=['red', 'orange', 'blue', 'lightgreen', 'green'])
plt.xticks(x_pos, x)
plt.title('Model Internals')
st.write(fig)


"""
## 2. Topic Classification
"""
results, prediction = model.get_topics(text)
scores = results['scores']
if scores[0]-scores[1] < 0.15:
    prediction = 'Others'
topic_scores = {k: v for k, v in zip(results['labels'], results['scores'])}
x = model.topic_labels
x_pos = np.arange(len(x))
y = [topic_scores.get(item) for item in x]

f"""
### Topic Verdict: {prediction}

Model Internals
"""
fig = plt.figure(figsize=(10, 5))
plt.bar(x_pos, y, color=['tab:blue', 'tab:cyan', 'tab:purple', 'tab:pink', 'tab:olive'])
plt.xticks(x_pos, x)
plt.title('Model Internals')
st.write(fig)



