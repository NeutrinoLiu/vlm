import argparse
import os
import json
from tqdm import tqdm
import torch
import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor
)
from peft import PeftModel
from threading import Thread
import time

from vision_process import process_vision_info

def load_model(args):

    # load tokenizer
    tokenizer_path = args.model_base if args.base_only else args.model_path
    print(f" >>> Loading tokenizer from {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # load image processor
    processor_path = args.model_base if args.base_only else args.model_path
    print(f" >>> Loading image processor from {processor_path}")
    full_processor = AutoProcessor.from_pretrained(
        processor_path,
        padding_side="left",
    )

    # load vanilla qwen2.5 model
    print(f" >>> Loading vanilla model from {args.model_base}")

    if not args.vllm:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_base,
            cache_dir=None,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16),
            device_map="auto",
        )
    else:
        raise NotImplementedError("vllm backend not implemented")
        assert args.base_only, "vllm does not support lora loading"
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )
        model = LLM(
            model=args.model_base,
            max_model_len=args.model_max_length,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if not args.base_only:
        # load finetuned non-lora part
        print(f" >>> Loading finetuned non-lora part from non_lora_trainables.bin")
        if os.path.exists(os.path.join(args.model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(args.model_path, 'non_lora_trainables.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        # load finetuned lora part
        print(f" >>> Loading finetuned lora part from {args.model_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        print(f"load model into {model.device}")

    return model, tokenizer, full_processor

def load_qa_pairs(qa_pair_path):
    with open(qa_pair_path, "r") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} QA pairs from {qa_pair_path}")
    return pairs

def prepare_ans_file(ans_path):
    if os.path.exists(ans_path):
        # overwrite = input(f"Answer file {ans_path} already exists. Overwriting it? (y/n): ")
        # if overwrite.lower() != "y":
        #     print("Exiting without overwriting.")
        #     return
        os.remove(ans_path)
    os.makedirs(os.path.dirname(ans_path), exist_ok=True)

def valid_txt(t):
    return t.strip() != ""

def parse_content(str_content, visual_type, visual_src):
    if not isinstance(visual_src, list):
        visual_src = [visual_src]
    tag_map = {
        "image": "<image>",
        "video": "<video>"
    }
    tag = tag_map[visual_type]
    ret = []

    assert str_content.count(tag) == len(visual_src), f"{str_content} has {str_content.count(tag)} {visual_type} tag, but {len(visual_src)} {visual_type} in the QA source"
    while tag in str_content:
        content_before, content_after = str_content.split(tag, 1)
        if valid_txt(content_before):
            ret.append({"type": "text", "text": content_before})
        ret.append({"type": visual_type, visual_type: "file://" + visual_src.pop(0)})
        if valid_txt(content_after):
            str_content = content_after
        else:
            return ret


def pair2msg(pair):
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    talk = [{"role": "system", "content": system_message}]
    for i, one_speak in enumerate(pair["conversations"]):
        try:
            if roles[one_speak["from"]] != roles["human"]:
                continue
        except:
            print(pair["conversations"])
    
        try:
            role = one_speak["role"]
            content = one_speak["content"]
        except:
            role = one_speak["from"]
            content = one_speak["value"]

        role = roles.get(role, role)
        if role == "user":
            if "image" in pair:
                content = parse_content(content, "image", pair["image"])
            elif "video" in pair:
                content = parse_content(content, "video", pair["video"])
        else:
            continue

        talk.append({"role": role, "content": content})
    return talk

def save_result(processor, ans_path, bs_buffer, inputs, generated_ids):
    # raw_output_list = processor.batch_decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )
    # print(raw_output_list)

    ans_ids_list = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    ans_text_list = processor.batch_decode(
        ans_ids_list,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    q_text_list = processor.batch_decode(
        inputs["input_ids"],
        skip_special_tokens=True)

    with open(ans_path, "a") as ans_file:
        for i in range(len(ans_text_list)):
            ans_text = ans_text_list[i].strip()
            q_text = q_text_list[i].strip()
            idx = bs_buffer[i]["id"]
            pair = bs_buffer[i]
            ans_file.write(json.dumps({
                "question_id": idx,
                "gt_ans": pair["conversations"][-1]["value"],
                "ans": ans_text,
                "question": q_text,
            }) + "\n")

def eval(args):

    pairs = load_qa_pairs(args.qa_pair)

    model, tokenizer, processor = load_model(args)
    prepare_ans_file(args.ans_path)

    bs_ctr = 0
    bs_buffer = []
    
    CHAT_TEMP = '''
    {% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
    '''
    down_sampled_pairs = pairs[::args.down_sample]
    for idx, pair in enumerate(tqdm(down_sampled_pairs)):
        bs_ctr += 1
        bs_buffer.append(pair)
        if bs_ctr < args.batch_size and not (idx == len(pairs) - 1):
            continue
        messages = [
            pair2msg(p) for p in bs_buffer
        ]
        # print(json.dumps(messages, indent=4))

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=CHAT_TEMP)
        images, videos = process_vision_info(messages)
        inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        save_result(
            processor,
            os.path.abspath(args.ans_path),
            bs_buffer,
            inputs,
            generated_ids,
        )

        bs_ctr = 0
        bs_buffer = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    model_path
    qa_pair
    ans_path
    model_max_length
    max_pixels
    min_pixels
    """
    parser.add_argument("--down-sample", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--qa-pair", type=str, default=None)
    parser.add_argument("--ans-path", type=str, default="./answers.json")
    parser.add_argument("--model-max-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=512*28*28)
    parser.add_argument("--min-pixels", type=int, default=786)
    parser.add_argument("--base-only", action="store_true", help="Whether to load lora model")
    parser.add_argument("--save-merged", action="store_true", help="save lora merged model")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--vllm", action="store_true", help="use vllm backend")
    args = parser.parse_args()

    if args.model_path is None:
        args.base_only = True
        print(f"model-path not set, use base-only model, will eval on {args.model_base}")
    print(" >>> Arguments:")
    print("   | min_pixels: ", args.min_pixels)
    print("   | max_pixels: ", args.max_pixels)
    print("   | model_max_length: ", args.model_max_length)
    print("   | max_new_tokens: ", args.max_new_tokens)
    print(" >>> plz confirm above is the same as the training time")

    eval(args)
    
