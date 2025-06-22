import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from qwenvl.data.rope2d import get_rope_index_25
from typing import Dict, Optional, Sequence, List

from PIL import Image
import math
import copy
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision import transforms as TF


def read_frame_paths_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # 去除每一行的换行符和空白字符
    frame_list = [line.strip() for line in lines if line.strip()]
    return frame_list


def resize_with_fixed_width_and_rounded_height(img, target_width=504, divisor=28):
    rounded_height = round(img.height * (target_width / img.width) / 28) * 28

    return img.resize((target_width, rounded_height), Image.BILINEAR)


def resize_tensor_with_fixed_width_and_rounded_height(tensor_img, target_width=518, divisor=14):
    """
    输入: tensor_img [C, H, W]，值范围 0~1 or 0~255
    输出: resized tensor_img [C, new_H, 518]，new_H 能被 divisor 整除
    """
    C, H, W = tensor_img.shape
    raw_height = H * target_width / W
    rounded_height = int((raw_height + divisor - 1) // divisor * divisor)

    tensor_img = tensor_img.unsqueeze(0)  # [1, C, H, W]
    resized = F.interpolate(
        tensor_img,
        size=(rounded_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [C, new_H, 518]

    return resized


def process_sampled_frames(sampled_image_list, processor, target_width=504):
    if len(sampled_image_list) == 0:
        raise ValueError("Empty image list")

    # 加载图像帧
    images = [Image.open(img_path).convert("RGB") for img_path in sampled_image_list]
    resized_images = [resize_with_fixed_width_and_rounded_height(img, target_width=target_width) for img in images]

    # images = [read_image(p).float() / 255.0 for p in sampled_image_list]
    # resized_images = [resize_tensor_with_fixed_width_and_rounded_height(img, target_width=518) for img in images]

    processor = copy.deepcopy(processor.image_processor)

    # 处理图像序列（假设支持视频输入）
    video_processed = processor.preprocess(
        images=None, videos=resized_images, return_tensors="pt"
    )

    video_tensor = video_processed["pixel_values_videos"]
    grid_thw = video_processed["video_grid_thw"][0]
    second_per_grid_ts = [
        1.0
    ] * len(grid_thw)

    # for vggt input
    to_tensor = TF.ToTensor()
    vggt_images = [to_tensor(img) for img in resized_images]
    vggt_images = torch.stack(vggt_images)
    
    return video_tensor, grid_thw, second_per_grid_ts, vggt_images


def build_qwen2_vl_prompt(
    source,
    tokenizer,
    grid_thw: List[int] = [],
    visual_type: str = "image",
) -> torch.Tensor:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{{ '<|im_start|>assistant\n' }}"  # 注意：末尾加 assistant prompt
    tokenizer.chat_template = chat_template

    # 构造对话结构（含视觉 token 插入）
    visual_replicate_index = 0
    new_messages = [{"role": "system", "content": system_message}]
    for conv in source:
        role = roles.get(conv.get("role", conv.get("from")), "user")
        content = conv.get("content", conv.get("value", ""))

        if role == "user" and f"<{visual_type}>" in content:
            parts = content.split(f"<{visual_type}>")
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                replacement = (
                    "<|vision_start|>"
                    + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index += 1
            new_parts.append(parts[-1])
            content = "".join(new_parts)

        new_messages.append({"role": role, "content": content})

    # 应用模板
    input_ids = tokenizer.apply_chat_template(
        new_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

    return input_ids