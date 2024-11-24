import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import os


class ReRouter:
    def __init__(self, model_dir="models/re_router/"):
        """
        Initialize the Re-Router.
        """
        os.makedirs(model_dir, exist_ok=True)
        self.vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        self.classifier_path = os.path.join(model_dir, "classifier.pkl")

    def train(self, data_path):
        """
        Train the Re-Router classifier using the provided CSV file.
        :param data_path: Path to the CSV file containing query and domain columns.
        """
        # Load the dataset
        data = pd.read_csv(data_path)
        X = data["query"]
        y = data["domain"]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train a Naive Bayes classifier
        classifier = MultinomialNB()
        classifier.fit(X_train_vec, y_train)

        # Save vectorizer and classifier
        with open(self.vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        with open(self.classifier_path, "wb") as f:
            pickle.dump(classifier, f)

        # Evaluate the classifier
        y_pred = classifier.predict(X_test_vec)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

    def predict(self, query):
        """
        Predict the domain of a given query.
        :param query: The user query as a string.
        :return: Predicted domain and confidence score.
        """
        # Load the saved vectorizer and classifier
        with open(self.vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(self.classifier_path, "rb") as f:
            classifier = pickle.load(f)

        # Predict the domain
        query_vec = vectorizer.transform([query])
        predicted_domain = classifier.predict(query_vec)[0]
        confidence_score = max(classifier.predict_proba(query_vec)[0])
        return predicted_domain, confidence_score
