
from templates.func import *
from templates.QA import QATemplate, QAMCTemplate
from templates.task import TaskSet, Hints
import json

def jsonfy(d):
    return f"```\n{json.dumps(d)}\n```"

class TrackingTasks(TaskSet):
    JSONFY = False

    @property 
    def COMMON_HINT(self):
        return f"""
        Lets patchify each frame into {self.H // self.patchsize_H} rows and {self.W // self.patchsize_W} columns, 
        and the origin is at the top-left corner of each frame. 
        We could index each patch by its row and col in the format of (row, col).
        Indexing range of row are of 0 ~ {self.H // self.patchsize_H - 1},
        indexing range of col are of 0 ~ {self.W // self.patchsize_W - 1}.
        You are an expert of multiview stereo. try your best to answer this question: 
        """.strip()

    @property
    def PATCH_HINT(self):
        return """
        The above frames are captured from the same 3D object or 3D scene yet from different viewpoints.
        """ + self.COMMON_HINT

    @property
    def PATCH_HINT_CONCAT(self):
        return """
        The above image is a horizontal-concat of two frames. 
        Those frames are captured from the same 3D object or 3D scene yet from different viewpoints.
        """ + self.COMMON_HINT

    @property
    def FORMAT_HINT(self):
        if self.JSONFY:
            return """
            Directly reply the patch index in the format of Json. For example {"row": xxx, "col": xxx}
            """
        else:
            return """
            Directly reply the patch index in the format of "row col", for example: 0 2
            """

    @property
    def FORMAT_HINT_FRAME(self):
        if self.JSONFY:
            return """
            Directly reply the patch indexes in each frame in the format of a Json list. For example:
            [{"frame": xxx, "row": xxx, "col": xxx}, ..., {"frame": xxx, "row": xxx, "col": xxx}]. 
            Frame index starts from 0. skip the frame if the object is not visible in that frame. 
            """
        else:
            return """
            Directly reply the patch indexes in each frame in the format of "frame row col, frame row col, ..."
            For example: 0 3 5, 1 7 9, 2 3 2
            Frame index starts from 0. skip the frame if the object is not visible in that frame.
            """

    def __init__(self,
                 captioner=None, basefilter=None, cfg={}, seed=0):
        super().__init__(cfg, seed)

        self.H = cfg["H"]
        self.W = cfg["W"]
        self.patchsize_H = cfg["patchsize_H"]
        self.patchsize_W = cfg["patchsize_W"]
        self.motion_thres = cfg["motion_thres"]
        self.num_frame = cfg["num_frame"]
        self.frame_strid = cfg["frame_stride"]

        obj_desc = obj_desc_fn if captioner is None else captioner.obj_desc_fn
        myfilter = basefilter if basefilter is not None else lambda x: True

        obj_cross_frame_tracking = QATemplate(
            Q_temp=self.PATCH_HINT +
            """
            We could notice that the <obj> is centered at patch <obj_grid> in the very first frame.
            Try to track the same object across all the following frames. Simply return the patch index of that object at each frame. 
            In your final answer also include the patch index of that object in the first frame.
            """
            + self.FORMAT_HINT_FRAME,
            A_temp="<patches_per_frame>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("obj_grid", self.first_frame_patch),
                ("patches_per_frame", self.all_frame_patches),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_frames": (self.num_frame, self.frame_strid, "samples stride"),
                "QA_type": "obj_cross_frame_tracking"
            },
        )
        self.tasks.update({
            "obj_cross_frame_tracking": obj_cross_frame_tracking,
        })

    def first_frame_patch(self, ctx):
        objs = ctx["objs"]
        frames = ctx["frames"]
        
        frame = frames[0]
        obj = objs[0]

        anno = anno_of_obj_from_frame(frame, obj)
        assert anno is not None, f"Object {obj['instance_token']} not found in frame {frame['sample_data_token']}"

        obj_center_x = (anno["2d_crop"]["diag"][0][0] + anno["2d_crop"]["diag"][0][1]) / 2
        obj_center_y = (anno["2d_crop"]["diag"][1][0] + anno["2d_crop"]["diag"][1][1]) / 2
        row = int(obj_center_y // self.patchsize_H)
        col = int(obj_center_x // self.patchsize_W)
        return f"({row}, {col})"

    def all_frame_patches(self, ctx):
        objs = ctx["objs"]
        frames = ctx["frames"]

        obj = objs[0]
        ret = []
        rows = []
        cols = []

        for idx, f in enumerate(frames):
            anno = anno_of_obj_from_frame(f, obj)
            if anno is None:
                continue
            obj_center_x = (anno["2d_crop"]["diag"][0][0] + anno["2d_crop"]["diag"][0][1]) / 2
            obj_center_y = (anno["2d_crop"]["diag"][1][0] + anno["2d_crop"]["diag"][1][1]) / 2
            row = int(obj_center_y // self.patchsize_H)
            col = int(obj_center_x // self.patchsize_W)
            ret.append({
                "frame": idx,
                "row": row,
                "col": col,
            })

            rows.append(row)
            cols.append(col)

        if len(rows) == 0 or max(rows) - min(rows) + max(cols) - min(cols) <= self.motion_thres:
            assert False, "object not moving, invalidate this qa"
        if self.JSONFY:
            return jsonfy(ret)
        else:
            return ", ".join([f"{r['frame']} {r['row']} {r['col']}" for r in ret])

