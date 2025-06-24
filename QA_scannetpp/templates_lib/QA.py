from dataclasses import dataclass
import os
import json
import random
from templates_lib.func import mc_json

@dataclass
class QAInstance:
    Q: str
    A: str
    ds: "QADataset"
    QA_type: str
    ctx: dict = None

    def dump(self):
        ctx_basic_keys = ["scene", "frames", "objs"]
        ctx_rest = {
            f"(ctx) {k}": self.ctx[k] for k in self.ctx.keys() if k not in ctx_basic_keys
        }
        ctx_basic = {
            "Question": self.Q,
            "Answer": self.A,
            "objs": [obj["instance_token"] for obj in self.ctx["objs"]],
            "timestamps": [frame['timestamp'] for frame in self.ctx["frames"]],
            "sample_data_tokens": [frame['sample_data_token'] for frame in self.ctx["frames"]],
            "scene": self.ctx["scene"].name,
            "dataset": self.ds.root,
            "QA_type": self.QA_type
        }
        return {**ctx_basic, **ctx_rest}

    def __str__(self):
        dump_dict = self.dump()
        return "\n".join(
            [f"{k}: {v}" for k, v in dump_dict.items()]
        )

    def qwen_format(self,
                    all_frames: bool = False,
                    preview: bool = False):

        if all_frames:
            # all frames, as a video
            images = self.ctx["scene"].frames_imgs
        else:
            # only roi frames, easier task
            images = [f["image_path"] for f in self.ctx["frames"]]
        
        # rel to abs
        images = [os.path.abspath(img) for img in images]
        if len(images) == 0:
            return {
                "conversations": [
                    {
                        "from": "human",
                        "value": self.Q
                    },
                    {
                        "from": "gpt",
                        "value": self.A
                    }
                ]
            }
        return {
            "image": images if not preview else (images[:1] + ["..."]),
            "conversations": [
                {
                    "from": "human",
                    "value": f"{self.Q}" + '\n<image>' * len(images)
                },
                {
                    "from": "gpt",
                    "value": self.A
                }
            ]
        }

class QAScene:
    IMG_NAMING = "CAM_FRONT_raw.jpg"
    META_NAMING = "CAM_FRONT_meta.json"

    def caption_ready(self, inst_token):
        return self.ds.caption_ready(self.name, inst_token)

    def objs_of_frame(self, frame_meta, obj_filter):
        """
        obj_filter: a function that takes in a frame_meta and returns a list of objects
        """
        objs = []
        for anno in frame_meta['annos']:
            if obj_filter(anno) and self.caption_ready(anno['instance_token']):
                new_obj = (anno['instance_token'], anno['category_name'])
                objs.append(new_obj)
        return objs

    def __init__(self, ds, name, sc_root: str, ):
        self.ds = ds
        self.name = name
        self.root = sc_root
        frames = os.listdir(sc_root)
        frames.sort()
        self.frames_imgs = []
        self.frames_metas = []
        self.objs = set()

        for i, frame in enumerate(frames):
            rel_path = os.path.join(sc_root, frame, QAScene.IMG_NAMING)
            abs_path = os.path.abspath(rel_path)
            self.frames_imgs.append(abs_path)
            js_file = os.path.join(sc_root, frame, QAScene.META_NAMING)
            with open(js_file, "r") as f:
                meta = json.load(f)
            meta['timestamp_idx'] = i
            self.frames_metas.append(meta)
            objs_in_frame = self.objs_of_frame(meta, lambda x: True)
            self.objs.update(objs_in_frame)
        
        self.objs = [
            {"instance_token": obj[0], "category_name": obj[1]} for obj in self.objs]

    def __getitem__(self, timestamp: str):
        for frame in self.frames_metas:
            if frame['timestamp'] == timestamp:
                return frame
        raise ValueError(f"Frame {timestamp} not found in scene {self.root}")

    def __len__(self):
        return len(self.frames_metas)

from tqdm.notebook import tqdm
from multiprocessing.pool import ThreadPool

def imap_with_tqdm(func, iterable, num_works=32):
    # multi thread
    with ThreadPool(num_works) as pool:
        return list(
                pool.imap(func, tqdm(iterable))
            )

class QADataset:
    def __init__(self, ds_root: str, cap_mgr=None):
        self.root = ds_root
        self.cap_mgr = cap_mgr
        self.scenes = os.listdir(ds_root)
        self.scenes = imap_with_tqdm(
                lambda x: QAScene(self, x, os.path.join(ds_root, x)),
                self.scenes
            )

    def caption_ready(self, scene, inst_token):
        if self.cap_mgr is None:
            return True
        return self.cap_mgr.caption_ready(scene, inst_token)

    def __getitem__(self, scene_token: str):
        for scene in self.scenes:
            if scene.root == scene_token:
                return scene
        raise ValueError(f"Scene {scene_token} not found in dataset")

    def __len__(self):
        return len(self.scenes)

class QATemplate:
    QA_SPLITTER = "<QASPLITTER>"

    def __init__(self, Q_temp: str, A_temp: str,  obj_mappers: list, obj_filter, config):
        self.QA_temp = f"{Q_temp}{QATemplate.QA_SPLITTER}{A_temp}"
        self.obj_mapper = obj_mappers
        self.obj_filter = obj_filter
        self.cfg = config

    def __call__(self, scene: QAScene, verbose: bool = False):
        # prepare resources
        num_objs = self.cfg["num_objs"]
        num_frames = self.cfg["num_frames"]
        if isinstance(num_frames, tuple):
            min_frame = num_frames[0]
            max_frame = num_frames[1]
            num_frames = random.randint(min_frame, max_frame)

        # randomly select n concecutive frames
        start_idx = random.randint(0, len(scene) - num_frames)
        frames = scene.frames_metas[start_idx:start_idx + num_frames]

        # randomly select n objects from the selected frames
        objs = set()
        for frame in frames:
            objs_in_frame = scene.objs_of_frame(frame, self.obj_filter)
            objs.update(objs_in_frame)
        objs = [{
            "instance_token": obj[0],
            "category_name": obj[1]
            } for obj in objs]
        objs = random.sample(objs, num_objs)


        # QA construction
        final_QA = self.QA_temp
        ctx = {
            "scene": scene,
            "frames": frames,
            "objs": objs
        }
        for kw, fn in self.obj_mapper:
            keyword = f"<{kw}>"
            assert keyword in final_QA, f"Keyword {keyword} not found in template {self.QA_temp}"
            # generate the text for the keyword,
            # also ctx might be updated
            final_QA = final_QA.replace(keyword, fn(ctx), 1)
        
        if verbose:
            print(f"scene: {scene.root}")
            print(f"frames: {[frame['timestamp'] for frame in frames]}")
            print(f"objs: {[obj['instance_token'] for obj in objs]}")

        Q = final_QA.split(self.QA_SPLITTER)[0]
        A = final_QA.split(self.QA_SPLITTER)[1]

        return QAInstance(
            Q=Q,
            A=A,
            ds=scene.ds,
            ctx=ctx,
            QA_type=self.cfg["QA_type"]
        )

class QAMCTemplate(QATemplate):
    OPTIONS = "ABCDEF"
    OPT_PREFIX = "MC_OPTIONS_PLACEHOLDER"
    ANS_PREFIX = "MC_ANSWER_PLACEHOLDER"

    def __init__(self, Q_temp: str, A_temp, obj_mappers: list, obj_filter, config):
        opt_num = config["num_options"]
        opt_mapper_gen = config["opt_mapper_gen"]
        ans_index_gen = config["ans_index_gen"]
        assert opt_num <= len(QAMCTemplate.OPTIONS), f"Answer number {opt_num} exceeds the number of options {len(QAMCTemplate.OPTIONS)}"
        options = "\n".join(
            [f"{QAMCTemplate.OPTIONS[i]}. <{QAMCTemplate.OPT_PREFIX}_{i}>" for i in range(opt_num)]
        )
        options = f"\n{options}\n"
        options_mappers = [
            (f"{QAMCTemplate.OPT_PREFIX}_{i}", opt_mapper_gen(i)) for i in range(opt_num)
        ]
        Q_temp_with_opts = Q_temp.replace(
            f"<{QAMCTemplate.OPT_PREFIX}>",
            options
        )

        def ans_mapper(ctx):
            idx = ans_index_gen(ctx)
            details = idx[1]
            idx = idx[0]
            assert idx < opt_num, f"Answer index {idx} exceeds the number of options {opt_num}"
            return mc_json(QAMCTemplate.OPTIONS[idx])
        
        merged_mappers = obj_mappers + options_mappers + [
            (QAMCTemplate.ANS_PREFIX, ans_mapper)
        ]
        super().__init__(Q_temp_with_opts, A_temp, merged_mappers, obj_filter, config)