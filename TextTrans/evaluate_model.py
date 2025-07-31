import pandas as pd
import numpy as np
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from evaluate import load  # ✅ Use evaluate instead of datasets
import torch

# === Load Model and Tokenizer ===
model_path = "./mt5_idiom_finetuned"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# === Load Test Data ===
test_df = pd.read_csv("idioms_test.csv")

# === Prepare Inputs ===
inputs = ["translate idiom to english: " + text for text in test_df["source"]]
labels = test_df["target"].tolist()

# === Tokenize Inputs ===
encodings = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(model.device)

# === Generate Translations ===
model.eval()
with torch.no_grad():
    outputs = model.generate(
        input_ids=encodings["input_ids"],
        attention_mask=encodings["attention_mask"],
        max_length=64,
        num_beams=4
    )

# === Decode Predictions ===
predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# === Evaluation Metrics (using evaluate) ===
bleu = load("bleu")
meteor = load("meteor")

bleu_score = bleu.compute(predictions=predictions, references=[[l] for l in labels])
meteor_score = meteor.compute(predictions=predictions, references=labels)

# === Output ===
print("=== Evaluation Metrics ===")
print(f"BLEU:   {bleu_score['bleu']:.4f}")
print(f"METEOR: {meteor_score['meteor']:.4f}")
print("\n=== Sample Predictions ===\n")

for i in range(min(10, len(test_df))):
    print(f"🔸 Source    : {test_df['source'][i]}")
    print(f"✅ Reference : {test_df['target'][i]}")
    print(f"🤖 Prediction: {predictions[i]}")
    print("-" * 50)
