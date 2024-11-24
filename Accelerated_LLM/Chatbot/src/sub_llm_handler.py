import os
import pickle
import numpy as np
from transformers import pipeline


class SubLLMHandler:
    def __init__(self):
        """
        Initialize the SubLLMHandler by loading necessary models and tools.
        """
        # Paths to the models
        re_router_model_dir = "models/re_router"
        fuzzy_logic_model_path = "models/fuzzy_logic/fuzzy_controller.pkl"

        # Load Re-Router model and vectorizer
        with open(os.path.join(re_router_model_dir, "classifier.pkl"), "rb") as f:
            self.re_router_model = pickle.load(f)
        with open(os.path.join(re_router_model_dir, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

        # Load Fuzzy Logic Controller
        with open(fuzzy_logic_model_path, "rb") as f:
            self.fuzzy_logic = pickle.load(f)

        # Load NLP pipelines for task-specific Sub-LLMs
        self.summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")
        self.translation_model = pipeline("translation_en_to_fr", model="t5-small")
        self.text_generation_model = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
        self.qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    def classify_query(self, query):
        """
        Classify the query into a domain using Re-Router and Fuzzy Logic Controller.
        :param query: User query as a string.
        :return: Classified domain and certainty (ambiguous or certain).
        """
        # Transform the query using the TF-IDF vectorizer
        query_vec = self.vectorizer.transform([query])

        # Predict domain and get confidence score
        domain_prediction = self.re_router_model.predict(query_vec)[0]
        confidence_score = max(self.re_router_model.predict_proba(query_vec)[0]) * 10

        # Compute ambiguity
        ambiguity_score = 10 - confidence_score

        # Use fuzzy logic to assess certainty
        self.fuzzy_logic.input["ambiguity"] = ambiguity_score
        self.fuzzy_logic.input["confidence"] = confidence_score
        self.fuzzy_logic.compute()
        domain_certainty = self.fuzzy_logic.output["domain_routing"]

        if domain_certainty < 5:
            return "ambiguous", "Ambiguous query. Please clarify."
        return domain_prediction, "certain"

    def handle_query(self, query):
        """
        Handle the query by routing it to the appropriate Sub-LLM.
        :param query: User query as a string.
        :return: Generated response or request for clarification.
        """
        domain, certainty = self.classify_query(query)
        if certainty == "Ambiguous query. Please clarify.":
            return certainty

        # Route query to the appropriate Sub-LLM
        if domain == "summarization":
            return self.summarize(query)
        elif domain == "translation":
            return self.translate(query)
        elif domain == "text_generation":
            return self.generate_text(query)
        elif domain == "qa":
            return self.answer_question(query)
        else:
            return "Domain not recognized."

    def summarize(self, text):
        """
        Perform text summarization using the summarization Sub-LLM.
        :param text: Text to be summarized.
        :return: Summarized text.
        """
        summary = self.summarization_model(text, max_length=50, min_length=20, do_sample=False)
        return summary[0]["summary_text"]

    def translate(self, text):
        """
        Translate text from English to French using the translation Sub-LLM.
        :param text: Text to be translated.
        :return: Translated text.
        """
        translation = self.translation_model(text)
        return translation[0]["translation_text"]

    def generate_text(self, prompt):
        """
        Generate creative text based on the prompt using the text generation Sub-LLM.
        :param prompt: Prompt for text generation.
        :return: Generated text.
        """
        generated = self.text_generation_model(prompt, max_length=100, num_return_sequences=1)
        return generated[0]["generated_text"]

    def answer_question(self, question):
        """
        Answer a question using the QA Sub-LLM.
        :param question: The question to answer.
        :return: Answer string.
        """
        # Define a basic context; in practice, this should be more extensive.
        context = (
            "Artificial Intelligence refers to the simulation of human intelligence in machines "
            "that are programmed to think and learn. AI applications include language processing, "
            "decision-making, and more."
        )
        answer = self.qa_model(question=question, context=context)
        return answer["answer"]
