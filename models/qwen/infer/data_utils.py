import os
import torch
import numpy as np
from typing import Dict, Optional, Sequence, List, Tuple
import copy
import transformers
from dataclasses import dataclass, field


from PIL import Image
try:
    from decord import VideoReader
except ImportError:
    print("missing decord, please install it with `pip install decord`")

from torch.utils.data import Dataset
from rope2d import get_rope_index_25

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
    ignore_gpt: bool = True,
    max_new = 1024
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{{ '<|im_start|>assistant\n' }}"  # 注意：末尾加 assistant prompt
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        talk = [{"role": "system", "content": system_message}]

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
            else:
                if ignore_gpt:
                    continue

            talk.append({"role": role, "content": content})

    ret = tokenizer.apply_chat_template(
        talk,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        padding="max_length",
        max_length=tokenizer.model_max_length - max_new,
        return_attention_mask=True
    )

    # print(f"key in input_ids {ret.keys()}")

    return ret

# @dataclass
# class TestDataArgs:

#     base_interval: int = field(default=1)
#     max_pixels: int = field(default=6 * 16 * 16 * 28 * 28)
#     min_pixels: int = field(default=28 * 28 * 16)
#     video_max_frames: Optional[int] = field(default=8)
#     video_min_frames: Optional[int] = field(default=4)
#     video_max_frame_pixels: int = field(default=6 * 16 * 16 * 28 * 28)
#     video_min_frame_pixels: int = field(default=4 * 28 * 28)

from train_args import DataArguments as TestDataArgs

# a simplified version of LazySupervisedDataset
class TestDataFeeder:
    def __init__(self, tokenizer, processor, args):
        self.data_args = TestDataArgs()
        # check if any args overwrite the default args (Namespace)
        for key, value in vars(args).items():
            if hasattr(self.data_args, key):
                setattr(self.data_args, key, value)
        self.processor = processor
        self.data_args.image_processor = processor.image_processor
        self.data_args.image_processor.max_pixels = self.data_args.max_pixels
        self.data_args.image_processor.min_pixels = self.data_args.min_pixels
        self.tokenizer = tokenizer
        self.get_rope_index = get_rope_index_25
        self.max_new_tokens = args.max_new_tokens

    def process_image(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def QA_to_ids(self, qa_pair) -> Dict[str, torch.Tensor]:
        if not isinstance(qa_pair, list):
            qa_pairs = [qa_pair]   # add batch dimension
        else:
            qa_pairs = qa_pair
        video = None
        if "image" in qa_pairs[0]:
            image_folder = qa_pairs[0]["data_path"] if "data_path" in qa_pairs[0] else "/"
            image_file = qa_pairs[0]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    results = [self.process_image(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]

            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            
            convs = copy.deepcopy([e["conversations"] for e in qa_pairs])
            data_dict = preprocess_qwen_2_visual(
                convs, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image",
                max_new = self.max_new_tokens
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
        elif "video" in qa_pairs[0]:
            video_file = qa_pairs[0]["video"]
            video_folder = qa_pairs[0]["data_path"] if "data_path" in qa_pairs[0] else "/"
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            convs = copy.deepcopy([e["conversations"] for e in qa_pairs])
            data_dict = preprocess_qwen_2_visual(
                convs, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video",
                max_new = self.max_new_tokens
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            convs = copy.deepcopy([e["conversations"] for e in qa_pairs])
            data_dict = preprocess_qwen_2_visual(
                convs, self.tokenizer, grid_thw=grid_thw_merged,
                max_new = self.max_new_tokens
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict_with_pose = data_dict.copy()
        data_dict_with_pose["position_ids"] = position_ids

        # print(f"qa_pairs {qa_pairs}")

        if "image" in qa_pairs[0]:
            data_dict_with_pose["pixel_values"] = image
            data_dict_with_pose["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "video" in qa_pairs[0]:
            data_dict_with_pose["pixel_values_videos"] = video
            data_dict_with_pose["video_grid_thw"] = grid_thw

        # data_dict includes:
        # input_ids, labels, position_ids, pixel_values, pixel_values_videos, image_grid_thw

        return data_dict_with_pose
