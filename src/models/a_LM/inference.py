# load model from hub
from peft import PeftModel, PeftConfig
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import ast 

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("nourijel/robotics_finetuned_text_perception")
model = PeftModel.from_pretrained(model, "nourijel/robotics_finetuned_text_perception")

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt = """[INST] Output only the keypoint location of the block corresponding to following instruction. Instructions are from the perspective of the black figure. Instruction:Pick up the blue block on your left, which is the second from the left nearest you. Image description: [{'name': 'blue', 'keypoint': {'x': 296, 'y': 552}}, {'name': 'yellow', 'keypoint': {'x': 331, 'y': 551}}, {'name': 'green', 'keypoint': {'x': 336, 'y': 512}}, {'name': 'orange', 'keypoint': {'x': 323, 'y': 485}}, {'name': 'blue', 'keypoint': {'x': 310, 'y': 454}}, {'name': 'yellow', 'keypoint': {'x': 352, 'y': 469}}, {'name': 'blue', 'keypoint': {'x': 395, 'y': 518}}, {'name': 'yellow', 'keypoint': {'x': 420, 'y': 548}}, {'name': 'orange', 'keypoint': {'x': 428, 'y': 444}}, {'name': 'blue', 'keypoint': {'x': 430, 'y': 486}}, {'name': 'blue', 'keypoint': {'x': 459, 'y': 457}}, {'name': 'blue', 'keypoint': {'x': 496, 'y': 459}}, {'name': 'yellow', 'keypoint': {'x': 481, 'y': 480}}, {'name': 'green', 'keypoint': {'x': 484, 'y': 500}}, {'name': 'green', 'keypoint': {'x': 498, 'y': 546}}][/INST]"""


model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

outputs = tokenizer.batch_decode(outputs)[0]


print(outputs)