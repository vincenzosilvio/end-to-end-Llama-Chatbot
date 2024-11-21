from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

def tokenize_function(examples, tokenizer):
    """ Tokenixe input and output"""
    return tokenizer(examples['input_text'], text_target = examples['response_text'], truncation=True)

def train_model(prepared_data_path, model_name, output_dir):
    """ Train the chatbot model"""
    df = pd.read_csv(prepared_data_path)

    train_val_split = int(0.8*len(df))
    train_data = Dataset.from_pandas(df.iloc[:train_val_split])
    val_data = Dataset.from_pandas(df.iloc[train_val_split:])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

    # Tokenize datasets
    tokenized_train = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val = val_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2,
    )

        # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

        # Train and save model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    train_model('prepared_data.csv', 'huggingface/llama-7b', './llama_chatbot_model')