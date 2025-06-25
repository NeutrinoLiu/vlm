import transformers
import torch
import os
import argparse
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor
)

def merge(args):

    # load tokenizer
    tokenizer_path = args.model_path
    print(f" >>> Loading tokenizer from {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=None,
        model_max_length=8192,
        padding_side="left",
        use_fast=False,
    )

    # load image processor
    processor_path = args.model_path
    print(f" >>> Loading image processor from {processor_path}")
    full_processor = AutoProcessor.from_pretrained(
        processor_path,
        padding_side="left",
    )

    # load vanilla qwen2.5 model
    print(f" >>> Loading vanilla model from {args.model_base}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_base,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16),
        device_map="auto",
    )

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

    merged_path = args.merged_path
    os.makedirs(merged_path, exist_ok=True)
    print(f" >>> Saving merged model to {merged_path}")
    model.save_pretrained(merged_path)
    full_processor.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--merged-path", type=str, default=None)
    args = parser.parse_args()
    assert args.model_path is not None
    if args.merged_path is None:
        base_name = os.path.basename(args.model_path)
        new_base_name = base_name + "_merged_qwen2.5" 
        args.merged_path = os.path.join(os.path.dirname(args.model_path), new_base_name)

    merge(args)