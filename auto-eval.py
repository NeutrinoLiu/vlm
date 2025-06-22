import json
import os
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("plan_file", type=str, default="./qa_plan.json")
args = parser.parse_args()

output_root = "/mnt/bn/nlhei-nas/liubangya/proj/vlm-found3d/tasks"

with open(args.plan_file, "r") as f:
    plan = json.load(f)

sft_json = "/mnt/bn/nlhei-nas/liubangya/proj/vlm-found3d/vlm/qwen/finetune-custom-dataset/custom.json"
base_model_default = "Qwen/Qwen2.5-VL-7B-Instruct"
eval_exec = "/mnt/bn/nlhei-nas/liubangya/proj/vlm-found3d/vlm/qwen/eval"
sft_exec = "/mnt/bn/nlhei-nas/liubangya/proj/vlm-found3d/vlm/qwen/finetune"

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
    run_name=None,
):
    os.system(f"cp {qa} {sft_json}")
    os.chdir(sft_exec)
    chkpt = os.path.join(output, f"chkpt_{get_timestamp()}")
    cmd = f"""
    python3 sft.py \
        master_port=22222
        lr=2e-5 \
        run_name={qa_name}_{get_timestamp()} \
        llm={base} \
        datasets=vlm_4o_custom \
        output_dir={chkpt}"""
    cmd = strip(cmd)
    print(cmd)
    ret = os.system(cmd)
    if ret!= 0:
        print(f"finetune failed: {qa}")
        return False
    return chkpt

for qa_name, qa_file in plan.items():
    print(f"{'=' * 10} eval: {qa_name}")

    task_path = f"{output_root}/{qa_name}"
    task_results = f"{task_path}/results"
    task_finetuned = f"{task_path}/finetuned"
    os.makedirs(task_path, exist_ok=True)
    os.makedirs(task_results, exist_ok=True)
    os.makedirs(task_finetuned, exist_ok=True)

    train_qa, test_qa, train_qa_meta, test_qa_meta = qa_file["train_qa"], qa_file["test_qa"], qa_file["train_qa_meta"], qa_file["test_qa_meta"]
    base_model = qa_file.get("base_model", base_model_default)

    print(f"{'=' * 3} base finetune")
    print(f"base model {base_model}")
    chkpt = finetune_qa(
        qa=train_qa,
        output=task_finetuned,
        base=base_model,
        run_name=qa_name,
    )
    if not chkpt:
        print(f"finetune failed: {qa_name}")
        break

    failed_flag = False

    def thread_base_eval():
        print(f"{'=' * 3} base eval")
        ret = eval_qa(
            qa=test_qa,
            meta=test_qa_meta,
            output=task_results,
            base=base_model,
            lora=None,
        )
        if not ret:
            failed_flag = True

    def thread_lora_eval():
        print(f"{'=' * 3} lora eval")
        ret = eval_qa(
            qa=test_qa,
            meta=test_qa_meta,
            output=task_results,
            base=base_model,
            lora=chkpt,
        )
        if not ret:
            failed_flag = True

    threads = []
    threads.append(threading.Thread(target=thread_base_eval))
    threads.append(threading.Thread(target=thread_lora_eval))
    for t in threads:
        t.start()
    # for t in threads:
        t.join()
    if failed_flag:
        print(f"eval failed: {qa_name}")
        break
    
    

