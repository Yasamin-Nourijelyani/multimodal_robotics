# source venv/bin/activate

# !pip install auto-gptq
# !pip install optimum
# !pip install bitsandbytes
# !pip uninstall torch -y
# !pip install torch==2.1

"""Fine tuning Script for LLM
Code is adapted from Shaw Talebi: https://colab.research.google.com/drive/1AErkPgDderPW0dgE230OOjEysd0QV1sR?usp=sharing
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

# Set the base directory for all Hugging Face caches
base_cache_directory = "/w/331/yasamin/.cache/huggingface"

# Setting environment variables for cache directories
os.environ["HF_HOME"] = base_cache_directory  # Sets the base cache directory for Hugging Face operations
os.environ["TRANSFORMERS_CACHE"] = os.path.join(base_cache_directory, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(base_cache_directory, "datasets")



# Now, your script continues as before, with the environment properly configured to use the specified cache directories.

# -----------load the model (fine tuned Mistral)-----------------
model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",  # This will automatically figure out the best use of CPU + GPU
                                             trust_remote_code=False,  # Prevents running custom model files on your machine
                                             revision="main")  # Specifies which version of the model to use

# ------------Loading the Tokenizer-----------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# -----------Preparing model for training--------------------------

model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)

#----------------- LoRA config----------------------
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()

# ------------- load dataset ------------------------

data = load_dataset("nourijel/robotics_perception_text")

# --------Tokenize function---------------

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ---------- Fine tuning ---------------
# hyperparameters
lr = 2e-4
batch_size = 32
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "robotics_finetuned_text_perception",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",

)

# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)


# train model
model.config.use_cache = False  # silence the warnings. re-enable for inference
trainer.train()

# renable warnings
model.config.use_cache = True