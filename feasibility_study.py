from transformers import pipeline


# Classes
class ZSL4FS:
    def __init__(self):
        self.model = pipeline("zero-shot-classification")
        self.sentiment_labels = ["Severe Negative", "Mild Negative", "Neutral", "Positive", "Severe Positive"]
        self.topic_labels = ["Financial Counselling", "Clinical Competency", "Waiting Time", "Medical Mismanagement", "Others"]

    def get_sentiment(self, text):
        result = self.model(text, candidate_labels=self.sentiment_labels)
        top_labels = result["labels"]
        if top_labels[0].endswith("Positive") and top_labels[1].endswith("Negative"):
             return result, "Neutral/Mixed"
        if top_labels[0].endswith("Negative") and top_labels[1].endswith("Positive"):
            return result, "Neutral/Mixed"
        sentiment = top_labels[0].split(" ")[-1]
        return result, sentiment

    def get_topics(self, text):
        result = self.model(text, candidate_labels=self.topic_labels)
        return result, result["labels"][0]


model = ZSL4FS()
