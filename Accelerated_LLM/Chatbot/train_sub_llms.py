from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict
from src.dataset_utils import prepare_dataset
import os


def fine_tune_sub_llm(model_name, task_name, data_path, output_dir):
    """
    Fine-tune a Sub-LLM for a specific task.
    :param model_name: Hugging Face model name (e.g., "facebook/bart-large-cnn").
    :param task_name: Task name (e.g., summarization, translation).
    :param data_path: Path to the dataset CSV file.
    :param output_dir: Directory to save the fine-tuned model.
    """
    # Validate dataset path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Prepare dataset with train and test splits
    dataset = prepare_dataset(data_path)

    # Debug: Check dataset splits
    print(f"Dataset for {task_name} successfully split:")
    print(f"Train examples: {len(dataset['train'])}, Test examples: {len(dataset['test'])}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Define tokenization function
    def preprocess_function(examples):
        inputs = examples["input"]
        outputs = examples["output"]

        # Tokenize input and output
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(outputs, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Tokenize train and test datasets
    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Train the model
    print(f"Starting training for {task_name}...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned for {task_name} and saved to {output_dir}")

if __name__ == "__main__":
    # Define task configurations
    tasks = [
    {
        "task_name": "summarization",
        "model_name": "facebook/bart-large-cnn",
        "data_path": r"C:\Users\nirmiti.deshmukh\Accelerated_LLM\Chatbot\Data\summarizel_train.csv",
        "output_dir": "models/sub_llms/summarization/",
    },
    {
        "task_name": "translation",
        "model_name": "t5-small",
        "data_path": r"C:\Users\nirmiti.deshmukh\Accelerated_LLM\Chatbot\Data\translation_train.csv",
        "output_dir": "models/sub_llms/translation/",
    },
    {
        "task_name": "text_generation",
        "model_name": "EleutherAI/gpt-neo-1.3B",
        "data_path": r"C:\Users\nirmiti.deshmukh\Accelerated_LLM\Chatbot\Data\text_Gen.csv",
        "output_dir": "models/sub_llms/text_generation/",
    },
    {
        "task_name": "qa",
        "model_name": "bert-large-uncased-whole-word-masking-finetuned-squad",
        "data_path": r"data/C:\Users\nirmiti.deshmukh\Accelerated_LLM\Chatbot\Data\qna_dataset.csv",
        "output_dir": "models/sub_llms/qa/",
    },
]

    # Train all Sub-LLMs
    for task in tasks:
        print(f"\nTraining Sub-LLM for {task['task_name']}...")
        fine_tune_sub_llm(
            model_name=task["model_name"],
            task_name=task["task_name"],
            data_path=task["data_path"],
            output_dir=task["output_dir"],
        )
