from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

def fine_tune_sub_llm(model_name, task_name, data_path, output_dir):
    """
    Fine-tune a sub-LLM for a specific task.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load dataset
    dataset = load_dataset("csv", data_files=data_path)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x["input"], truncation=True, padding="max_length"), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
    )

    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    trainer.train()
    model.save_pretrained(output_dir)
