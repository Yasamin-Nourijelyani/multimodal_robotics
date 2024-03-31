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

prompt = """Output only the keypoint location dictionary value, nothing else of the block corresponding to following instruction. Instructions are from the perspective you looking forward at the table of blocks with their keypoints and colors given. Instruction:Pick up the green block that is right next to the blue block. {'width': 756, 'height': 660, 'annotations': [{'id': '8f941e4d-8e6b-4e37-a2cd-f19b3dbb1dda', 'keypoint': {'x': 293.1651, 'y': 452.9717}, 'name': 'orange'}, {'id': '77f8b560-0c54-4923-9c65-8e8973d5cc1a', 'keypoint': {'x': 293.9434, 'y': 485.6604}, 'name': 'orange'}, {'id': 'c2fdf254-12fa-43ee-9d04-fd097d6a2fad', 'keypoint': {'x': 332.8585, 'y': 517.5708}, 'name': 'orange'}, {'id': 'ee022624-310e-4867-8360-cd61e6a5e8ec', 'keypoint': {'x': 325.0755, 'y': 557.2642}, 'name': 'orange'}, {'id': 'ca78545f-1817-447e-a767-1ec186e40025', 'keypoint': {'x': 377.2217, 'y': 478.6557}, 'name': 'orange'}, {'id': 'b7ea891b-3095-4df8-a718-b193f34c851c', 'keypoint': {'x': 386.5613, 'y': 492.6651}, 'name': 'orange'}, {'id': '54b85951-9b7d-4a22-9b94-5856c8f4cc7e', 'keypoint': {'x': 441.8208, 'y': 499.6698}, 'name': 'orange'}, {'id': 'ef2b7cbb-fa7e-4894-8e50-310d43a6f3eb', 'keypoint': {'x': 385.0047, 'y': 448.3019}, 'name': 'yellow'}, {'id': 'ab7d8b08-4eaf-41cf-8066-9aba376bb546', 'keypoint': {'x': 392.0094, 'y': 523.0189}, 'name': 'yellow'}, {'id': 'aec059f3-2def-4458-9ea4-3fa5916cab5b', 'keypoint': {'x': 466.7264, 'y': 442.8538}, 'name': 'yellow'}, {'id': '39fe094d-1e93-4828-ae7e-d672aeb88451', 'keypoint': {'x': 356.2075, 'y': 533.9151}, 'name': 'blue'}, {'id': '57cde523-a685-45a5-b623-1b11057840da', 'keypoint': {'x': 487.7406, 'y': 488.7736}, 'name': 'blue'}, {'id': 'da91e8aa-4087-4689-95ed-80e4ba9a93ec', 'keypoint': {'x': 327.4104, 'y': 472.4292}, 'name': 'green'}, {'id': '9406ceb5-4de6-4915-8da6-132b32721d7c', 'keypoint': {'x': 508.7547, 'y': 511.3443}, 'name': 'green'}, {'id': 'fd93334e-be71-4598-9c11-abb001661b87', 'keypoint': {'x': 497.0802, 'y': 560.3774}, 'name': 'green'}, {'id': '3067ae97-d698-4a21-82f6-e94a33ed51a0', 'keypoint': {'x': 380.3349, 'y': 59.1509}, 'name': 'you'}]}
"""


model.eval()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

outputs = tokenizer.batch_decode(outputs)[0]


coords_pattern = r"\{'x':\s*[-+]?\d*\.?\d+,\s*'y':\s*[-+]?\d*\.?\d+\}</s>"
matches = re.findall(coords_pattern, outputs)

print(ast.literal_eval(matches[0][:-4]))