import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import json

class InvalidQAContext(Exception):
    def __init__(self, func_name):
        super().__init__(f"Invalid context for function {func_name}.")

def anno_of_obj_from_frame(frame, obj):
    token = obj["instance_token"]
    for anno in frame["annos"]:
        if anno["instance_token"] == token:
            return anno
    return None

from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

def rp(obj, ndigits=2):
    # reduced precision
    if isinstance(obj, float):
        return round(obj, ndigits)
    elif isinstance(obj, dict):
        return {k: rp(v, ndigits) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [rp(elem, ndigits) for elem in obj]
    else:
        return obj

dist_json = lambda x: "```json\n" + json.dumps({"dist": rp(x)}) + "\n```"
xy_json = lambda x, y: "```json\n" + json.dumps({"x": rp(x), "y": rp(y)}) + "\n```"
mc_json = lambda mc: "```json\n" + json.dumps({"ans": mc}) + "\n```"
def json_unwrap(s):
    if s.startswith("```json"):
        s = s[7:]
    if s.endswith("```"):
        s = s[:-3]

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        print(f"fail to parse json: {s}")
        raise

    return obj
# --------------------------- physicals calculation -------------------------- #

dist = lambda pt1, pt2: ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5
dist_tolerate = lambda pt1, pt2 : dist(pt1, pt2) if dist(pt1, pt2) > 0.1 else 0.0

class Captioner:
    def __init__(self, cap_file):
        with open(cap_file, 'r') as f:
            self.caps = yaml.safe_load(f)
    def obj_desc_fn(self, idx):
        def obj_desc(ctx):
            objs = ctx['objs']
            scene = ctx['scene'].name
            assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
            obj = objs[idx]
            cap = self.caps[scene][obj["instance_token"]]
            cap = cap.replace("'", "").replace('"', '')
            cap = f'"{cap}"'
            return cap
        return obj_desc
    def caption_ready(self, scene, inst_token):
        return scene in self.caps and inst_token in self.caps[scene]

def obj_desc_fn(idx):
    def obj_desc(ctx):
        objs = ctx['objs']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        return objs[idx]["category_name"].split(".")[-1]
    return obj_desc

def obj_cam_dist_fn(idx):
    def obj_cam_dist(ctx):
        objs = ctx['objs']
        frames = ctx['frames']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        assert len(frames) == 1, f"obj_cam_dist should only have one frame, but got {len(frames)}"
        frame = frames[0]
        obj = objs[idx]

        anno = anno_of_obj_from_frame(frame, obj)
        assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dist = dist(cam_t, obj_t)
        return dist_json(cam_dist)
    return obj_cam_dist

def obj_dist_between(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 2, f"obj_dist_between should only have two objects, but got {len(objs)}"
    assert len(frames) == 1, f"obj_dist_between should only have one frame, but got {len(frames)}"
    frame = frames[0]
    obj1 = objs[0]
    obj2 = objs[1]

    anno1 = anno_of_obj_from_frame(frame, obj1)
    anno2 = anno_of_obj_from_frame(frame, obj2)
    assert anno1 is not None, f"Object {obj1['instance_token']} not found in frame {frame['sample_data_token']}"
    assert anno2 is not None, f"Object {obj2['instance_token']} not found in frame {frame['sample_data_token']}"

    obj1_t = anno1["box_t"]
    obj2_t = anno2["box_t"]
    between_dist = dist(obj1_t, obj2_t)
    return dist_json(between_dist)

def obj_cam_dist_minmax(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 1, f"obj_cam_dist_range should only have one object, but got {len(objs)}"
    assert len(frames) > 1, f"obj_cam_dist_range should have at least two frames, but got {len(frames)}"
    obj = objs[0]
    if ctx['minmax'] == 'min':
        fn = min
    elif ctx['minmax'] == 'max':
        fn = max
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")
    cam_dists = []
    for frame in frames:
        anno = anno_of_obj_from_frame(frame, obj)
        if anno is None: continue
        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dists.append(
            dist(cam_t, obj_t)
        )

    return dist_json(fn(cam_dists))

def obj_dist_between_minmax(ctx):
    objs = ctx['objs']
    frames = ctx['frames']

    assert len(objs) == 2, f"obj_dist_between_range should only have two objects, but got {len(objs)}"
    assert len(frames) > 1, f"obj_dist_between_range should have at least two frames, but got {len(frames)}"
    obj1 = objs[0]
    obj2 = objs[1]
    between_dists = []
    if ctx['minmax'] == 'min':
        fn = min
    elif ctx['minmax'] == 'max':
        fn = max
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")
    for frame in frames:
        anno1 = anno_of_obj_from_frame(frame, obj1)
        anno2 = anno_of_obj_from_frame(frame, obj2)
        if anno1 is None or anno2 is None: continue
        obj1_t = anno1["box_t"]
        obj2_t = anno2["box_t"]
        between_dists.append(
            dist(obj1_t, obj2_t)
        )
    
    if len(between_dists) == 0:
        raise InvalidQAContext("obj_dist_between_minmax")

    return dist_json(fn(between_dists))

# ----------------------- cross frame movement related ----------------------- #

def ego_movement_calc(ctx):
    frames = ctx['frames']
    assert len(frames) > 1, f"ego_movement_calc should have at least two frames, but got {len(frames)}"
    ego_t0 = frames[0]["cam_t"]
    ego_t1 = frames[-1]["cam_t"]

    movement = dist_tolerate(ego_t0, ego_t1)
    return dist_json(movement)

def obj_movement_fn(idx):
    def obj_movement(ctx):
        objs = ctx['objs']
        frames = ctx['frames']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        assert len(frames) > 1, f"obj_movement should have at least two frames, but got {len(frames)}"
        obj = objs[idx]
        obj_t0 = None
        obj_t1 = None
        for frame in frames:
            anno = anno_of_obj_from_frame(frame, obj)
            if anno is None: continue
            if obj_t0 is None:
                obj_t0 = anno["box_t"]
            else:
                obj_t1 = anno["box_t"]
                break

        if obj_t0 is None or obj_t1 is None:
            raise InvalidQAContext("obj_movement, object not found in frames")

        movement = dist_tolerate(obj_t0, obj_t1)
        return dist_json(movement)
    return obj_movement


# coords related

def local_coords(anno, frame):
    cam_quat = np.array(frame["cam_r"]) # w, x, y, z
    cam_pose = np.array(frame["cam_t"])
    obj_pos = np.array(anno["box_t"])

    rel_pos = obj_pos - cam_pose

    r = R.from_quat(cam_quat, scalar_first=True)
    r_inv = r.inv()

    ret = r_inv.apply(rel_pos)
    x = - ret[1] # x axis is ->right
    y = ret[0]  # y axis is  ->forward
    return x, y

def obj_local_coords_fn(idx):
    def obj_local_coords(ctx):
        objs = ctx['objs']
        frames = ctx['frames']
        assert len(objs) > idx, f"query object index {idx} out of range, only {len(objs)} objects"
        assert len(frames) == 1, f"obj_local_coords should only have one frame, but got {len(frames)}"
        frame = frames[0]
        obj = objs[idx]

        anno = anno_of_obj_from_frame(frame, obj)
        assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

        x, y = local_coords(anno, frame)
        return xy_json(x, y)

    return obj_local_coords

# ---------------------------- indexing functions ---------------------------- #

def index_of_minmax_fn(val_fn):
    def index_of_minmax(ctx):
        objs = ctx['objs']
        if ctx['minmax'] == 'min':
            minmax_fn = min
        elif ctx['minmax'] == 'max':
            minmax_fn = max
        else:
            raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")

        assert len(objs) > 1, f"index_of_minmax should have multiple objects, but got {len(objs)}"

        val = []
        for idx, obj in enumerate(objs):
            val_fn_impl = val_fn(idx)
            val.append(val_fn_impl(ctx))
        
        idx_of_minmax = val.index(minmax_fn(val))
        return idx_of_minmax, val
    return index_of_minmax

def index_of_minmax_dist(ctx):
    objs = ctx['objs']
    frames = ctx['frames']
    if ctx['minmax'] == 'min':
        fn = min
        place_holder = float("inf")
    elif ctx['minmax'] == 'max':
        fn = max
        place_holder = float("-inf")
    else:
        raise ValueError(f"minmax should be 'min' or 'max', but got {ctx['minmax']}")

    assert len(objs) > 1, f"index_of_minmax_dist should have multiple objects, but got {len(objs)}"
    assert len(frames) == 1, f"index_of_minmax_dist should only have one frame, but got {len(frames)}"

    frame = frames[0]
    cam_dists = []
    for obj in objs:
        anno = anno_of_obj_from_frame(frame, obj)
        if anno is None: 
            cam_dists.append(place_holder)
            continue
        cam_t = frame["cam_t"]
        obj_t = anno["box_t"]
        cam_dists.append(
            dist(cam_t, obj_t)
        )
    
    idx_of_minmax = cam_dists.index(fn(cam_dists))
    if cam_dists[idx_of_minmax] == place_holder:
        raise InvalidQAContext("index_of_minmax_dist")
    return idx_of_minmax, cam_dists

def index_of_roulette(ctx):
    assert "roulette_correct_idx" in ctx, "roulette should be in context"
    return ctx["roulette_correct_idx"], None

# ------------------------------- frame related ------------------------------ #

def frame_idx_fn(idx, from_zero=False):
    def frame_ts(ctx):
        frames = ctx['frames']
        assert len(frames) > idx, f"query frame index {idx} out of range, only {len(frames)} frames"
        return str(frames[idx]["timestamp_idx"]) if not from_zero else "0"
    return frame_ts

def frame_range_fn(from_zero=False):
    def frame_range(ctx):
        frames = ctx['frames']

        assert len(frames) > 0, "frame_range should have at least one frame"

        if from_zero:
            return f"0 ~ {len(frames) - 1}"
        return f"{frames[0]['timestamp_idx']} ~ {frames[-1]['timestamp_idx']}" 
    return frame_range

# ------------------------------ str generation ------------------------------ #

def minmax(min_prompt="minimal", max_prompt="maximal"):
    minmax = random.choice(['min', 'max'])
    def minmax_fn(ctx):
        ctx['minmax'] = minmax
        return f"{min_prompt if minmax == 'min' else max_prompt}"
    return minmax_fn

# ------------------------ turn metrics qa into mc qa ------------------------ #

def roulette(gt_fn, max_opts=5, perturb=0.15):
    def roulette_option_gen(idx):
        if idx == 0:
            # init ctx with correct_idx "roulette_correct_idx"
            # init ctx with all_options "roulette_options"
            def roulette_init(ctx):
                correct_ans = gt_fn(ctx)
                # dont have to be json format when displaying options
                if isinstance(correct_ans, str):
                    correct_ans = json_unwrap(correct_ans)
                correct_idx = random.randint(0, max_opts - 1)
                ctx["roulette_correct_idx"] = correct_idx
                ctx["roulette_options"] = [None] * max_opts
                for i in range(max_opts):
                    shift = (i - correct_idx) * perturb
                    dummy_ans = {
                        k: v * (1 + shift) if isinstance(v, float) else v for k, v in correct_ans.items()
                    }
                    dummy_ans = rp(dummy_ans)
                    dummy_ans = ", ".join([str(v) for _, v in dummy_ans.items()])
                    ctx["roulette_options"][i] = dummy_ans
                return ctx["roulette_options"][0]
            return roulette_init
        else:
            def roulette_fetch(ctx):
                assert "roulette_options" in ctx, "roulette should be in context"
                return ctx["roulette_options"][idx]
            return roulette_fetch
    return roulette_option_gen
