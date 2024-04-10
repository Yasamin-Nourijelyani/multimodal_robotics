from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
save_directory = "models/a_LM/model_directory"  # Choose your directory

# Download and save the model
model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

model = PeftModel.from_pretrained(model, "nourijel/robotics_finetuned_text_perception")

    
model.save_pretrained(save_directory)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer have been saved to {save_directory}")
    