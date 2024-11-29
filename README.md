# Accelerated LLM System

## Overview

This repository contains the implementation of an **Accelerated Large Language Model (LLM)** system, which integrates multiple advanced components to intelligently classify, route, and handle user queries. The system includes:

- **Re-Router**: A classifier that predicts the domain of a user query.
- **Fuzzy Logic**: A controller that assesses the certainty of a query.
- **Sub-LLMs**: Task-specific models for summarization, translation, text generation, and question answering.

The system is designed to be scalable and flexible, handling a variety of user queries across different domains.

---

## Components

### 1. **Re-Router**

The **Re-Router** is responsible for classifying the domain of a user query. It uses the **Naive Bayes** classifier to predict the domain, leveraging **TF-IDF vectorization** to process query data.

- **Key Features**:
  - Uses **TF-IDF** to convert text data into numerical features.
  - Classifies queries into predefined domains (e.g., summarization, translation).
  - Saves trained vectorizer and classifier for future use.

### 2. **Fuzzy Logic Controller**

The **Fuzzy Logic Controller** evaluates the **ambiguity** and **confidence** of a query, determining how certain or ambiguous the model's prediction is.

- **Key Features**:
  - Uses **fuzzy membership functions** to assess query uncertainty.
  - Computes **domain certainty** based on the Re-Router’s confidence score.
  - Returns results like "certain" or "ambiguous."

### 3. **Sub-LLMs (Task-Specific Models)**

The **Sub-LLMs** handle specific tasks based on the domain predicted by the Re-Router. These tasks include summarization, translation, text generation, and question answering.

- **Key Features**:
  - **Summarization**: Uses **facebook/bart-large-cnn**.
  - **Translation**: Uses **t5-small** for English-to-French translation.
  - **Text Generation**: Uses **GPT-Neo 1.3B** for creative text generation.
  - **QA**: Uses **BERT** for question answering.

Each model is fine-tuned for its respective task using custom datasets.

---

## Installation

### Dependencies

To run the system, install the following dependencies:

```bash
pip install transformers scikit-learn pandas skfuzzy deap
**### Setup**
Clone this repository:
```bash
git clone https://github.com/yourusername/Accelerated_Large_Language_Model.git
cd accelerated-llm
Set up the data directory with labeled query data for training the Re-Router and datasets for the Sub-LLMs.
Run the training scripts for each component.


### Model Training
1. Re-Router Training
The Re-Router is trained on a dataset of queries labeled with domains. The model uses TF-IDF vectorization and a Naive Bayes classifier for domain classification.
The model's performance is evaluated using classification metrics like accuracy and the classification report.
2. Sub-LLM Fine-Tuning
Each Sub-LLM is fine-tuned on task-specific datasets. These models (e.g., BART for summarization, T5 for translation) are fine-tuned using Hugging Face Trainer on the provided datasets.
3. Fuzzy Logic Controller Training
The Fuzzy Logic Controller is trained with predefined rules and fuzzy membership functions, assessing query ambiguity and confidence.
4. Evolutionary Neural Network Training
Evolutionary algorithms are used to optimize the architecture of a neural network for query domain classification.
Query Handling Flow
The system processes incoming queries through the following pipeline:

Domain Classification: The query is passed to the Re-Router to predict its domain (e.g., summarization, translation).
Certainty Evaluation: The Fuzzy Logic Controller evaluates the confidence and ambiguity of the prediction.
Task Routing: The query is routed to the appropriate Sub-LLM (Summarization, Translation, Text Generation, or QA).
Response Generation: The Sub-LLM processes the query and returns a generated response.
Evaluation
1. Re-Router
The accuracy and classification reports are used to evaluate the model’s performance in domain classification.
2. Sub-LLMs
Task-specific models are evaluated using metrics like BLEU score (for translation), ROUGE score (for summarization), and accuracy for QA.
3. Fuzzy Logic
The Fuzzy Logic Controller is evaluated based on its ability to correctly classify queries as "certain" or "ambiguous."
