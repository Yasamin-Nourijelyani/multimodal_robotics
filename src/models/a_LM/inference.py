from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import re 
import ast
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo



tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


model.eval() # model in evaluation mode (dropout modules are deactivated)

# craft prompt
comment = " Image description: [{'name': 'blue', 'keypoint': {'x': 296, 'y': 552}}, {'name': 'yellow', 'keypoint': {'x': 331, 'y': 551}}, {'name': 'green', 'keypoint': {'x': 336, 'y': 512}}, {'name': 'orange', 'keypoint': {'x': 323, 'y': 485}}, {'name': 'blue', 'keypoint': {'x': 310, 'y': 454}}, {'name': 'yellow', 'keypoint': {'x': 352, 'y': 469}}, {'name': 'blue', 'keypoint': {'x': 395, 'y': 518}}, {'name': 'yellow', 'keypoint': {'x': 420, 'y': 548}}, {'name': 'orange', 'keypoint': {'x': 428, 'y': 444}}, {'name': 'blue', 'keypoint': {'x': 430, 'y': 486}}, {'name': 'blue', 'keypoint': {'x': 459, 'y': 457}}, {'name': 'blue', 'keypoint': {'x': 496, 'y': 459}}, {'name': 'yellow', 'keypoint': {'x': 481, 'y': 480}}, {'name': 'green', 'keypoint': {'x': 484, 'y': 500}}, {'name': 'green', 'keypoint': {'x': 498, 'y': 546}}]"
prompt=f'''[INST] {comment} [/INST]'''

# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

#print(tokenizer.batch_decode(outputs)[0])




intstructions_string = f""" Output only the keypoint location of the block corresponding to following instruction. Instructions are from the perspective of the black figure. Instruction:Pick up the blue block on your left, which is the second from the left nearest you. """

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

prompt = prompt_template(comment)
#print(prompt)



# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")



# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)
print("_______________not tuned__________________")
print(tokenizer.batch_decode(outputs)[0])




print("___________Fine tuned________________")


# load model from hub


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("nourijel/robotics_finetuned_text_perception")
model = PeftModel.from_pretrained(model, "nourijel/robotics_finetuned_text_perception")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


intstructions_string = f""" Output only the keypoint location of the block corresponding to following instruction. Instructions are from the perspective of the black figure. Instruction:Pick up the blue block on your left, which is the second from the left nearest you."""

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

comment = "Image description: [{'name': 'blue', 'keypoint': {'x': 296, 'y': 552}}, {'name': 'yellow', 'keypoint': {'x': 331, 'y': 551}}, {'name': 'green', 'keypoint': {'x': 336, 'y': 512}}, {'name': 'orange', 'keypoint': {'x': 323, 'y': 485}}, {'name': 'blue', 'keypoint': {'x': 310, 'y': 454}}, {'name': 'yellow', 'keypoint': {'x': 352, 'y': 469}}, {'name': 'blue', 'keypoint': {'x': 395, 'y': 518}}, {'name': 'yellow', 'keypoint': {'x': 420, 'y': 548}}, {'name': 'orange', 'keypoint': {'x': 428, 'y': 444}}, {'name': 'blue', 'keypoint': {'x': 430, 'y': 486}}, {'name': 'blue', 'keypoint': {'x': 459, 'y': 457}}, {'name': 'blue', 'keypoint': {'x': 496, 'y': 459}}, {'name': 'yellow', 'keypoint': {'x': 481, 'y': 480}}, {'name': 'green', 'keypoint': {'x': 484, 'y': 500}}, {'name': 'green', 'keypoint': {'x': 498, 'y': 546}}]"
prompt = prompt_template(comment)
#print(prompt)


model.eval()


inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print("______________ex 1_____________________")
text = tokenizer.batch_decode(outputs)[0]
print(text)
dict_string = re.search(r"\[/INST\] (.*?)\[/s\]", text).group(1)
extracted_dict = ast.literal_eval(dict_string)
extracted_dict
print("__________dict_______________")
print(type(extracted_dict))

print(extracted_dict)


# ex2
comment = "Image description: [{'name': 'blue', 'keypoint': {'x': 334, 'y': 563}}, {'name': 'yellow', 'keypoint': {'x': 321, 'y': 526}}, {'name': 'orange', 'keypoint': {'x': 336, 'y': 503}}, {'name': 'blue', 'keypoint': {'x': 292, 'y': 470}}, {'name': 'yellow', 'keypoint': {'x': 318, 'y': 450}}, {'name': 'blue', 'keypoint': {'x': 375, 'y': 430}}, {'name': 'orange', 'keypoint': {'x': 376, 'y': 506}}, {'name': 'blue', 'keypoint': {'x': 360, 'y': 538}}, {'name': 'green', 'keypoint': {'x': 399, 'y': 475}}, {'name': 'green', 'keypoint': {'x': 407, 'y': 531}}, {'name': 'orange', 'keypoint': {'x': 440, 'y': 426}}, {'name': 'blue', 'keypoint': {'x': 463, 'y': 456}}, {'name': 'yellow', 'keypoint': {'x': 454, 'y': 536}}, {'name': 'orange', 'keypoint': {'x': 492, 'y': 525}}, {'name': 'orange', 'keypoint': {'x': 409, 'y': 442}}]"
prompt = prompt_template(comment)
                                                                                                                                                                                                                
                                                                                                                                                                                                                 
model.eval()
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
print("___________________ex 2________________________")
text = tokenizer.batch_decode(outputs)[0]
print(text)
dict_string = re.search(r"\[/INST\] (.*?)\[/s\]", text).group(1)
extracted_dict = ast.literal_eval(dict_string)
extracted_dict
print("__________dict_______________")
print(type(extracted_dict))

print(extracted_dict)
