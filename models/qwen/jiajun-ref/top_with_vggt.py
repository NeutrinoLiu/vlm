import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.mm_utils import get_model_name_from_path

from PIL import Image
import math

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_constructor import *
from vila3r.model.builder import load_vila3r_model


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_frame_paths_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # 去除每一行的换行符和空白字符
    frame_list = [line.strip() for line in lines if line.strip()]
    return frame_list


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    model = load_vila3r_model(model_path, args.vggt_path, args.model_base, model_name)

    processor = AutoProcessor.from_pretrained(args.model_base)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    # with open(args.question_file, 'r') as file:
    #     questions = json.load(file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        # scene_id = line["scene_id"]
        video_file = line["scene_id"]
        video_path = os.path.join(args.video_folder, "posed_images", video_file)
        sample_info_file = os.path.join(args.video_folder, "sample_32_frames", f"{video_file}.txt")

        sampled_frames = read_frame_paths_from_txt(sample_info_file)
        video, grid_thw, second_per_grid_ts, vggt_images = process_sampled_frames(sampled_frames, processor, target_width=args.max_image_size)
        video = [video]

        cur_prompt = line["text"]
        cur_prompt = cur_prompt + " Answer the question using one word or one phrase."

        conversations = [
            {
                "from": "human",
                "value": "<video>\n" + cur_prompt
            },
        ]

        grid_thw_merged = copy.deepcopy(grid_thw)
        if not isinstance(grid_thw, Sequence):
            grid_thw_merged = [grid_thw_merged]
            grid_thw = [grid_thw]
        grid_thw_merged = [
            merged_thw.prod() // processor.image_processor.merge_size**2
            for merged_thw in grid_thw_merged
        ]
        # sources = copy.deepcopy([conversations])

        input_ids = build_qwen2_vl_prompt(
                        conversations,
                        tokenizer=tokenizer,
                        grid_thw=grid_thw_merged,
                        visual_type="video"
                    )
        
        position_ids, _ = get_rope_index_25(
            processor.image_processor.merge_size,
            input_ids,
            video_grid_thw=torch.stack(grid_thw, dim=0),
            second_per_grid_ts=second_per_grid_ts,
        )

        data_dict = dict(
            input_ids=input_ids.to(device="cuda"),
            position_ids=position_ids.to(device="cuda"),
        )
        
        data_dict["pixel_values_videos"] = torch.stack(video, dim=0).to(dtype=torch.bfloat16, device="cuda")
        data_dict["video_grid_thw"] = torch.stack(grid_thw, dim=0).to(device="cuda")
        data_dict["vggt_images"] = vggt_images.to(dtype=torch.bfloat16, device="cuda").unsqueeze(0)

        inputs = data_dict

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": output_text,
                                   "answer_id": ans_id,
                                   "model_id": "Qwen2_5_VL_3B",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="playground/data/LLaVA-3D-Pretrain")
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_image_size", type=int, default=504)
    parser.add_argument("--vggt_path", type=str, default="./checkpoints/vggt/model.pt")
    args = parser.parse_args()

    eval_model(args)
