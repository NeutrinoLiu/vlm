import os
import json
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
import argparse
import logging

warnings.filterwarnings("ignore")
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_model_and_processor(model_dir):
    """
    Load the VLM model and processor.
    """
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto"
    )
    return model, processor


def inference_batch(image_paths, prompts, model, processor, max_new_tokens):
    """
    Perform batch inference for a list of image paths and corresponding prompts.

    Args:
      image_paths: List of image file paths.
      prompts: List of text prompts corresponding to each image.
      model: Loaded VLM model.
      processor: Loaded processor.
      max_new_tokens:

    Returns:
      List of model-generated output texts for each image-text pair in the batch.
    """
    # Prepare the messages in the required format for batch inference
    messages_batch = [
       [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]
        for image_path, prompt in zip(image_paths, prompts)
    ]

    # Prepare the input for the processor
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    image_inputs, video_inputs = process_vision_info(messages_batch)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Perform batch inference for all images
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts