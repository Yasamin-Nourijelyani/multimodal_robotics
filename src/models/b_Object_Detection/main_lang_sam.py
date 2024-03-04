# https://lightning.ai/pages/community/lang-segment-anything-object-detection-and-segmentation-with-text-prompt/


from  PIL  import  Image
from lang_sam import LangSAM
from utils import draw_image

model = LangSAM()
image_pil = Image.open('../assets/car.jpeg').convert("RGB")
text_prompt = 'car, wheel'
masks, boxes, labels, logits = model.predict(image_pil, text_prompt)
image = draw_image(image_pil, masks, boxes, labels)
