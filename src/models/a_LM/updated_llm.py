from huggingface_hub import login


"""Fine tuning Script for LLM
Code is adapted from Shaw Talebi: https://colab.research.google.com/drive/1AErkPgDderPW0dgE230OOjEysd0QV1sR?usp=sharing
"""




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



    # ---------- Fine tuning ---------------

def fine_tune(output_dir, lr=2e-4, batch_size=64, num_epochs=10):    
    """Run to fine tune the TheBloke/Mistral-7B-Instruct-v0.2-GPTQ model"""


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

    # tokenize training and validation datasets
    tokenized_data = data.map(tokenize_function, batched=True)

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
    model, trainer = fine_tune(output_dir, lr=2e-4, batch_size=32, num_epochs=10)


    print("------- put on hf-------")
    write_key = '...'# TODO: add hf key from https://huggingface.co/settings/tokens
    login(write_key)

    hf_name = 'nourijel' # your hf username or org name
    model_id = hf_name + "/" + "robotics_finetuned_text_perception"

    model.push_to_hub(model_id)
    trainer.push_to_hub(model_id)

    # # load model from hub
    # model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    # model = AutoModelForCausalLM.from_pretrained(model_name,
    #                                             device_map="auto",
    #                                             trust_remote_code=False,
    #                                             revision="main")

    # config = PeftConfig.from_pretrained("nourijel/robotics_finetuned_text_perception")
    # model = PeftModel.from_pretrained(model, "nourijel/robotics_finetuned_text_perception")

    # # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    # # ------ evaluation-----

    # intstructions_string = """Output only the keypoint location of the block corresponding to following instruction,
    #   nothing else. Instructions are from the perspective you looking forward at the table of blocks with their 
    #   keypoints and colors given. Instruction:Pick up the green block that is directly behind the farthest yellow block from you. 
    #         {'width': 756, 'height': 660, 'annotations': [{'id': '8beb9a9e-a451-450a-b594-7a9ae18e74f9', 'keypoint': {'x': 458.9434, 'y': 531.5802},
    #  'name': 'orange'}, {'id': 'ee956837-2183-4b52-8aa2-dfa990f1d376', 'keypoint': {'x': 386.5613, 'y': 529.2453}, 'name': 'yellow'},
    #    {'id': '427a5df9-dc11-494e-8632-81ffcd5e66c8', 'keypoint': {'x': 471.3962, 'y': 453.75}, 'name': 'yellow'}, 
    #    {'id': 'bcc302a4-c191-4bc2-82e0-1ceedd4a2928', 'keypoint': {'x': 329.7453, 'y': 453.75}, 'name': 'blue'}, 
    #    {'id': '8584b004-32b4-47bd-9100-a2eab16c5301', 'keypoint': {'x': 294.7217, 'y': 501.2264}, 'name': 'blue'}, 
    #    {'id': 'ba4430e4-6293-47e3-b512-0e78897c96fb', 'keypoint': {'x': 317.2925, 'y': 520.684}, 'name': 'blue'}, 
    #    {'id': '9c57d65f-59e8-4d45-a882-721a56841969', 'keypoint': {'x': 433.2594, 'y': 498.1132}, 'name': 'blue'}, 
    #    {'id': '0e241c7c-e225-4266-9044-a65451aceb68', 'keypoint': {'x': 361.6557, 'y': 428.8443}, 'name': 'green'}, 
    #    {'id': '050252d9-484a-4a87-a46c-1f69f5d7ea14', 'keypoint': {'x': 416.9151, 'y': 435.8491}, 'name': 'green'}, 
    #    {'id': '5b9ab273-0e91-4589-9a86-d2969adc4e30', 'keypoint': {'x': 435.5943, 'y': 445.967}, 'name': 'green'},
    #      {'id': '4b452968-0027-4b94-92e6-e8dcd32d1e25', 'keypoint': {'x': 468.283, 'y': 489.5519}, 'name': 'green'}, 
    #      {'id': '3ab98d12-5e5c-4bdf-931d-55f2ee2d23b2', 'keypoint': {'x': 386.5613, 'y': 498.8915}, 'name': 'green'},
    #        {'id': '416f60f7-59aa-46b3-858c-7dfde5e23875', 'keypoint': {'x': 353.0943, 'y': 497.3349}, 'name': 'green'}, 
    #        {'id': 'ce8faeb7-52eb-4dc8-8db9-4295f1b077e3', 'keypoint': {'x': 329.7453, 'y': 491.8868}, 'name': 'green'},
    #          {'id': '6ea4f70c-aa80-4f14-a1ac-d26081a22a0d', 'keypoint': {'x': 379.5566, 'y': 52.1462}, 'name': 'you'}]}"""
   

    # model.eval()
    # inputs = tokenizer(intstructions_string, return_tensors="pt")

    # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    # print(tokenizer.batch_decode(outputs)[0])
