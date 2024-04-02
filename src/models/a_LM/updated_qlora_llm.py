

"""Fine tuning Script for LLM
Code is adapted from Shaw Talebi: https://colab.research.google.com/drive/1AErkPgDderPW0dgE230OOjEysd0QV1sR?usp=sharing
"""




# --------Tokenize function---------------

# create tokenize function
def tokenize_function(examples):
    # extract text and tokenize it
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



    # ---------- Fine tuning ---------------

def fine_tune(output_dir, model, lr=2e-4, batch_size=64, num_epochs=10):    
    """Run to fine tune the TheBloke/Mistral-7B-Instruct-v0.2-GPTQ model"""
    

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

    #data = load_dataset("nourijel/text_only")
    data_path_train = "../../../data/train_test_data/train.jsonl"
    data_train = load_dataset("json", data_files=data_path_train)
    # tokenize training and validation datasets
    tokenized_data_train = data_train.map(tokenize_function, batched=True)
  

    data_path_test = "../../../data/train_test_data/test.jsonl"
    data_test = load_dataset("json", data_files=data_path_test)
    # tokenize training and validation datasets
    tokenized_data_test = data_test.map(tokenize_function, batched=True)


    # setting pad token
    tokenizer.pad_token = tokenizer.eos_token
    # data collator
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)





    # define training arguments
    training_args = transformers.TrainingArguments(
        output_dir= output_dir,
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
        train_dataset=tokenized_data_train["train"],
        eval_dataset=tokenized_data_test["train"],
        args=training_args,
        data_collator=data_collator
    )


    # train model
    model.config.use_cache = False  # silence the warnings. re-enable for inference
    trainer.train()

    # renable warnings
    model.config.use_cache = True

    return model, trainer




if __name__ == "__main__":

    
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import prepare_model_for_kbit_training, PeftModel, PeftConfig
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    import transformers
    from huggingface_hub import login


    output_dir = "robotics_finetuned_text_perception"

    # -----------load the model (fine tuned Mistral)-----------------
    model_name="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",  # This will automatically figure out the best use of CPU + GPU
                                                trust_remote_code=False,  # Prevents running custom model files on your machine
                                                revision="main")  # Specifies which version of the model to use

    # ------------Loading the Tokenizer-----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model, trainer = fine_tune(output_dir, model, lr=2e-4, batch_size=32, num_epochs=10)


    print("------- put on hf-------")
    write_key = '...'# TODO: add hf key from https://huggingface.co/settings/tokens
    login(write_key)

    hf_name = 'nourijel'
    model_id = hf_name + "/" + "robotics_finetuned_text_perception"

    model.push_to_hub(model_id)
    trainer.push_to_hub(model_id)


