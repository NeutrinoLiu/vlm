
import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor
)
import matplotlib.pyplot as plt
import glob
import random
import sys
# sys.path.append('/home/nuo/vlm_proj/nuscenes-devkit/python-sdk/')
# from nuscenes.nuscenes import NuScenes

import numpy as np
import json
import yaml
import os

# model_id = "Ertugrul/Qwen2.5-VL-7B-Captioner-Relaxed"
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# DATA_VER = 'v1.0-mini'
# DATA_ROOT = './nuscenes-data'
# nusc = NuScenes(version=DATA_VER, dataroot=DATA_ROOT, verbose=True)

# the model requires more than 16GB of VRAM, 
# if you don't have you can use bitsandbytes to quantize the model to 8bit or 4bit


from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
  model_id,
  device_map="auto",
  trust_remote_code=True, 
  torch_dtype=torch.bfloat16,
  attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU or use "eager" for older GPUs
)

# you can change the min and max pixels to fit your needs to decrease compute cost to trade off quality
min_pixels = 256*28*28
max_pixels = 1280*28*28

processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels, min_pixels=min_pixels)

def generate_description(path, model, processor, prompt):
    system_message = "You are an expert image labeller."

    image_inputs = Image.open(path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_inputs},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    
    # min_p and temperature are experemental parameters, you can change them to fit your needs
    generated_ids = model.generate(**inputs, max_new_tokens=512, min_p=0.1, do_sample=True, temperature=1.5)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


 
def validate(caption):
    bad_words = ["photo", "image", "picture", "figure", "silhouette", "scene"]
    for word in bad_words:
        if word in caption:
            print(f"bad word: {word}")
            return False
    return True

def captioning(img_path, prompt, verbose=False):
    instance_token = img_path.split("/")[-1].split(".")[0]
    # nusc_instance = nusc.get('instance', instance_token)
    # category_name = nusc.get('category', nusc_instance['category_token'])['name']
    # hint_mapper = {
    #     "human.pedestrian.adult": "adult",
    #     "human.pedestrian.child": "child",
    #     "human.pedestrian.construction_worker": "construction worker",
    #     "human.pedestrian.police_officer": "police officer",
    #     "human.pedestrian.stroller": "stroller",
    #     "vehicle.bicycle": "bicycle",
    #     "vehicle.bus.bendy": "bendy bus",
    #     "vehicle.bus.rigid": "bus",
    #     "vehicle.car": "car",
    #     "vehicle.motorcycle": "motorcycle",
    #     "vehicle.trailer": "trailer",
    #     "vehicle.truck": "truck"
    # }
    # if category_name in hint_mapper:
    #     hint = hint_mapper[category_name]
    # else:
    #     hint = category_name.split(".")[-1]

    hint = instance_token.split("_")[4]
    print(f"hint: {hint}, img_path: {img_path}")

    prompt = prompt.replace("<hint>", hint)
    

    img = Image.open(img_path)
    if img.size[0] < 28 or img.size[1] < 28:
        print(f"image too small: {img_path}")
        return None

    if verbose:
        try:
            plt.close()
        except:
            pass
        print(f"hint: {hint}")
        plt.axis('off')
        plt.imshow(img)

    description = generate_description(img_path, model, processor, prompt)

    # grey = img.convert("L")
    # grey_pixels = np.array(grey)
    # avg_brightness = grey_pixels.mean()
    # darkness = 1 - (avg_brightness / 255)
    # print(f"darkness: {darkness:.2f}")

    # return(description)

    cap = description.lower().strip()
    if cap.startswith("a "):
        cap = description[2:]
    elif cap.startswith("an "):
        cap = description[3:]

    # remove all quotes
    cap = cap.replace("'", "").replace('"', '')
    cap = cap.split(".")[0]

    if not cap.startswith("the"):
        cap = "the " + cap

    if not validate(cap):
        print(f"invalid caption: {cap}")
        return None
    
    return cap

def cap_choice(
    caps, 
    img,
    prompt,
    verbose = False,
):
    letters = "ABCDEFGH"[:len(caps)]
    letter_and_option = [
        (letter, option) for letter, option in zip(letters, caps)
    ]
    options = [
        f"{option}" for letter, option in letter_and_option
    ]
    option_prompt = "\n".join(options)
    full_prompt = f"{prompt}\n{option_prompt}"
    if verbose:
        print(f"prompt: {full_prompt}")
    description = generate_description(img, model, processor, full_prompt)

    cap = description.lower().strip()
    if cap.startswith("a "):
        cap = description[2:]
    elif cap.startswith("an "):
        cap = description[3:]

    # remove all quotes
    cap = cap.replace("'", "").replace('"', '').replace("**", "")
    cap = cap.split(".")[0].split("reasoning")[0]

    if not cap.startswith("the"):
        cap = "the " + cap

    return cap
    
    


from tqdm import tqdm

root = "./structured-data-crops"
dump_path = "./captions.yaml"
scenes = glob.glob(f"{root}/*")

try:
    with open(dump_path, "r") as f:
        history = yaml.safe_load(f)
except FileNotFoundError:
    history = {}

CAP_ALL = True
num_retry = 3

prompt = "Provide a short noun phrase captioning this <hint> from a self driving dataset"
"such as 'the black sedan with red logo' or 'the man in a blue t-shirt and jeans'. "
"following such template: the {color} {object type} {extra description} ."
"MUST include the color of the object."

prompt = (
    "Provide a short noun phrase captioning this <hint> from an indoor 3D reconstruction dataset, "
    "such as 'the black office chair with armrests' or 'the brown wooden cabinet with two drawers'. "
    "following such template: the {color} {object type} {extra description}. "
    "MUST include the color of the object."
)


# "simply describe the object, do not include 'image', 'photo', 'picture', 'scene' or 'silhouette'. "
# "TEN words max. MUST describe COLORS. "
# "if it is too dark, just say 'unknown'. "

choice_prompt = "Which of the following captions closet and most concisely describes the object shown in the img? NOTE: must select from one of the below, return the caption only."


if CAP_ALL:
    for sc in tqdm(scenes):
        images = glob.glob(f"{sc}/*")
        sc_name = sc.split("/")[-1]
        if sc_name in history:
            print(f"Already processed {sc_name}, skipping...")
            continue
        if len(images) == 0:
            print(f"No images found in the directory: {sc}")
            continue
        print(f"Processing {len(images)} images in {sc_name}...")
        sc_objs = {}
        for image_path in images:
            obj = image_path.split("/")[-1].split(".")[0]

            caps = [
                captioning(image_path, prompt=prompt) for _ in range(num_retry) 
            ]
            caps = [cap for cap in caps if cap is not None]
            cap = cap_choice(caps, image_path, choice_prompt) if len(caps) > 1 else None

            if cap is None:
                print(f"Failed to caption image: {image_path}")
                continue
            sc_objs[obj] = cap

        with open(dump_path, "a") as f:
            yaml.dump({sc_name: sc_objs}, f, default_flow_style=False, width=float("inf"))
else:
    sc = random.choice(scenes)
    images = glob.glob(f"{sc}/*")
    image = random.choice(images)
    caps = [
        captioning(image, verbose=True, prompt=prompt) for _ in range(num_retry) 
    ]
    caps = [cap for cap in caps if cap is not None]
    best = cap_choice(caps, image, choice_prompt) if len(caps) > 1 else None
    print(f"ans: {best}")




