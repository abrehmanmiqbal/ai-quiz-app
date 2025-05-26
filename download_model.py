# download_model.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "ramsrigouthamg/t5_squad_v1"
save_path = "t5_model"  # This will create a local folder

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Save to local folder
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model and tokenizer downloaded to ./t5_model!")
