import json
import os
import threading
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("plan_file", type=str, default="./qa_plan.json")
args = parser.parse_args()

output_root = "/mnt/bn/nlhei-nas/liubangya/proj/vlm/workspace"

with open(args.plan_file, "r") as f:
    plan = json.load(f)

sft_json = "/mnt/bn/nlhei-nas/liubangya/proj/vlm/models/tmp/qwen-custom-dataset/custom.json"
model_base_default = "Qwen/Qwen2.5-VL-7B-Instruct"
eval_exec = "/mnt/bn/nlhei-nas/liubangya/proj/vlm/models/qwen/infer"
sft_exec = "/mnt/bn/nlhei-nas/liubangya/proj/vlm/models/qwen/finetune"

"""
{"qa_task_name":{
    "train_qa" : "path",
    "test_qa" : "path",
    "train_qa_meta" : "path",
    "test_qa_meta" : "path",
}}
"""
def strip(s):
    return " ".join(s.split())

def eval_qa(
    qa,
    meta,
    output,
    base,
    lora=None,
    batch_size=16,
):
    os.chdir(eval_exec)
    if lora is None:
        ans_file = os.path.join(output, "ans_base.json")
        cmd = f"""
        python3 eval.py \
            --model-path None \
            --model-base {base} \
            --qa-pair {qa} \
            --ans-path {ans_file} \
            --batch-size {batch_size} \
            --base-only"""
        cmd = strip(cmd)
    else:
        ans_file = os.path.join(output, "ans_lora.json")
        assert os.path.exists(lora)
        cmd = f"""
        python3 eval.py \
            --model-path {lora} \
            --model-base {base} \
            --qa-pair {qa} \
            --ans-path {ans_file} \
            --batch-size {batch_size}"""
        cmd = strip(cmd)
    print(cmd)
    ret = os.system(cmd)
    if ret!= 0:
        print(f"eval failed: {qa}")
        return False
    return True

import datetime
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def finetune_qa(
    qa,
    output,
    base,
    extra_args,
    run_name=None,
):
    # os.system(f"cp {qa} {sft_json}")
    shutil.copy(qa, sft_json)
    os.chdir(sft_exec)
    chkpt = os.path.join(output, f"chkpt")
    cmd = f"""
    export WANDB_API_KEY=b8de1ba7e50c8756b94a6ca7497e8e50b6c25830 &&
    python3 sft.py \
        master_port=22222
        lr=2e-5 \
        run_name={qa_name} \
        llm={base} \
        datasets=vlm_4o_custom \
        output_dir={chkpt}"""
    extra_args = " ".join([f"{k}={v}" for k, v in extra_args.items()])
    cmd += f" {extra_args}"
    cmd = strip(cmd)
    print(cmd)
    ret = os.system(cmd)
    if ret!= 0:
        print(f"finetune failed: {qa}")
        return False
    return chkpt

# main

for qa_name, qa_file in plan.items():
    print(f"{'=' * 10} eval: {qa_name}")

    task_path = f"{output_root}/{qa_name}"
    task_results = f"{task_path}/results"
    task_finetuned = f"{task_path}/finetuned"
    os.makedirs(task_path, exist_ok=True)
    os.makedirs(task_results, exist_ok=True)
    os.makedirs(task_finetuned, exist_ok=True)

    train_qa, test_qa, train_qa_meta, test_qa_meta = (
        qa_file.get("train_qa", None),
        qa_file.get("test_qa", None),
        qa_file.get("train_qa_meta", None),
        qa_file.get("test_qa_meta", None)
        )
    task_pairs_path = f"{task_path}/pairs"
    os.makedirs(task_pairs_path, exist_ok=True)
    # copy qa pair if not exists
    for file in [train_qa, test_qa, train_qa_meta, test_qa_meta]:
        if file is None:
            continue
        filename = os.path.basename(file)
        if not os.path.exists(f"{task_pairs_path}/{filename}"):
            shutil.copy(file, f"{task_pairs_path}/{filename}")

    model_base = qa_file.get("model_base", model_base_default)

    print(f"{'=' * 3} base finetune")
    print(f"base model {model_base}")
    chkpt = finetune_qa(
        qa=train_qa,
        output=task_finetuned,
        base=model_base,
        run_name=qa_name,
        extra_args = qa_file.get("train_args", {})
    )
    if not chkpt:
        print(f"finetune failed: {qa_name}")
        break

    failed_flag = False

    def thread_base_eval():
        global failed_flag
        print(f"{'=' * 3} base eval")
        ret = eval_qa(
            qa=test_qa,
            meta=test_qa_meta,
            output=task_results,
            base=model_base,
            lora=None,
        )
        if not ret:
            failed_flag = True

    def thread_lora_eval():
        global failed_flag
        print(f"{'=' * 3} lora eval")
        ret = eval_qa(
            qa=test_qa,
            meta=test_qa_meta,
            output=task_results,
            base=model_base,
            lora=chkpt,
        )
        if not ret:
            failed_flag = True

    threads = []
    if qa_file.get("eval_finetune", False):
        threads.append(threading.Thread(target=thread_lora_eval))
    if qa_file.get("eval_base", False):
        threads.append(threading.Thread(target=thread_base_eval))

    if qa_file.get("parallel_eval", False):
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for t in threads:
            t.start()
            t.join()
    if failed_flag:
        print(f"eval failed: {qa_name}")
        break
    
    

