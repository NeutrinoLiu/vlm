 
import json
import os
import random

os.chdir("./")
print(f"current working dir", os.getcwd())

# function pool here
from templates_lib.filter import *
from templates_lib.func import *
from templates_lib.QA import QADataset


# ### Static::Measurement::object_distance

 
DS_ROOT = "./structured-data"
# DS_ROOT = "/home/nuo/vlm_proj/QA_scannetpp/structured-data"

OUTPUT_DIR = "pairs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_QWEN = f"{OUTPUT_DIR}/QA_pairs_qwen.json"
OUTPUT_JSON = f"{OUTPUT_DIR}/QA_pairs.json"
TEST_SPLIT = 0.2
CAP_FILE = "captions.yaml"

random.seed(0)
myCap = Captioner(CAP_FILE)
ds = QADataset(DS_ROOT, myCap)



 
from templates_lib.task import MultiTaskSet
from templates_lib.task.distance_tasks import DistTasks
from templates_lib.task.movement_tasks import MovementTasks

total_qas = 50_000

tasks_cfg = {
    "roi_frame_only": True,
}

myfilter = filter_all(
    filter_visiblity,
    filter_area,
    black_list_fn([
            "movable_object.trafficcone",
            "movable_object.barrier",
        ])
    )

taskset = MultiTaskSet(
    subsets=[DistTasks, MovementTasks],
    captioner=myCap,
    basefilter=myfilter,
    cfg=tasks_cfg)

qas, stats = taskset.produce(
    dataset=ds,
    num_qas=total_qas,
    # verbose=True
)

print(f"total {len(qas)} qas")
print(f"stats: {json.dumps(stats, indent=2)}")

all_dumps = [qa.dump() for qa in qas]
content_stats = {
    "objs": set(),
    "scenes": set(),
}
for qa in all_dumps:
    content_stats["objs"].update(qa["objs"])
    content_stats["scenes"].update([qa["scene"]])
print(f"total objects: {len(content_stats['objs'])}")
print(f"total scenes: {len(content_stats['scenes'])}")



 

num_test = int(len(qas) * TEST_SPLIT)
qas_train = qas[:-num_test]
qas_test = qas[-num_test:]

all_frames = not taskset.cfg["roi_frame_only"]

with open(OUTPUT_QWEN.replace(".", ".test."), "w") as f:
    qas_dumps = [qa.qwen_format(all_frames=all_frames) for qa in qas_test]
    for i, qa in enumerate(qas_dumps):
        qa["id"] = i
    json.dump(
        qas_dumps, f, indent=2
    )
with open(OUTPUT_QWEN.replace(".", ".train."), "w") as f:
    qas_dumps = [qa.qwen_format(all_frames=all_frames) for qa in qas_train]
    for i, qa in enumerate(qas_dumps):
        qa["id"] = i
    json.dump(
        qas_dumps, f, indent=2
    )
with open(OUTPUT_JSON.replace(".", ".test."), "w") as f:
    qas_dumps = [qa.dump() for qa in qas_test]
    for i, qa in enumerate(qas_dumps):
        qa["id"] = i
    json.dump(qas_dumps, f, indent=2)
with open(OUTPUT_JSON.replace(".", ".train."), "w") as f:
    qas_dumps = [qa.dump() for qa in qas_train]
    for i, qa in enumerate(qas_dumps):
        qa["id"] = i
    json.dump(qas_dumps, f, indent=2)



