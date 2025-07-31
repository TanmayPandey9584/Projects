import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import numpy as np
import torch
import os

# === Load Better Idiom Dataset ===
train_df = pd.read_csv("idioms_train.csv")
test_df = pd.read_csv("idioms_test.csv")

train_df["source"] = "translate idiom to english: " + train_df["source"]
test_df["source"] = "translate idiom to english: " + test_df["source"]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# === Model ===
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# === Preprocess ===
def preprocess(batch):
    inputs = tokenizer(batch["source"], padding="max_length", truncation=True, max_length=64)
    labels = tokenizer(text_target=batch["target"], padding="max_length", truncation=True, max_length=64)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# === Train Config ===
output_dir = "./mt5_idiom_finetuned"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
