import streamlit as st
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch

# === Load Fine-tuned Model ===
model_path = "./mt5_idiom_finetuned"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

# === Streamlit UI ===
st.set_page_config(page_title="Idiom Translator", layout="centered")
st.title("🧠 Cultural Idiom Translator")
st.markdown("Translate Hindi idioms to English with cultural context understanding.")

idiom_input = st.text_input("📝 Enter a Hindi idiom:")

if idiom_input:
    with st.spinner("Translating..."):
        prompt = "translate idiom to english: " + idiom_input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64,
                num_beams=4
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    st.success("✅ Translation")
    st.markdown(f"**{decoded}**")

    with st.expander("🔍 Debug Info"):
        st.text(f"Input IDs: {inputs['input_ids']}")
        st.text(f"Raw Output: {outputs}")
