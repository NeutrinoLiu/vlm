
from templates.func import *
from templates.QA import QATemplate, QAMCTemplate
from templates.task import TaskSet, Hints
import json

def jsonfy(d):
    return f"```json\n{json.dumps(d)}\n```"

class GridIndexTasks(TaskSet):

    @property 
    def COMMON_HINT(self):
        return f"""
        Lets patchify each frame into {self.H // self.patchsize_H} rows and {self.W // self.patchsize_W} columns, 
        and the origin is at the top-left corner of each frame. 
        We could index each patch by its row and col in the format of (row, col).
        Indexing range of row are of 0 ~ {self.H // self.patchsize_H - 1},
        indexing range of col are of 0 ~ {self.W // self.patchsize_W - 1}.
        """.strip()

    @property 
    def FORMAT_HINT(self):
        return """
        Directly reply the patch index in the format of Json. For example {"row": xxx, "col": xxx}
        """.strip()

    def __init__(self, 
        H, W, patchsize_H, patchsize_W,
        captioner=None, basefilter=None, cfg={}, seed=0):
        super().__init__(cfg, seed)

        self.H = H
        self.W = W
        self.patchsize_H = patchsize_H
        self.patchsize_W = patchsize_W

        myfilter = basefilter if basefilter is not None else lambda x: True

        grid_indexing = QATemplate(
            Q_temp=self.COMMON_HINT +
            """
            I have masked out one of the patch in the following image with a red square.
            could you tell me what is the index of that patch?
            """
            + self.FORMAT_HINT,
            A_temp="<patch_idx>",
            obj_mappers=[
                ("patch_idx", self.rand_patch_idx),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 0,
                "num_frames": 1,
                "QA_type": "grid_indexing"
            },
        )


        self.tasks.update({
            "grid_indexing": grid_indexing,
        })

    def rand_patch_idx(self, ctx):
        row = random.randint(0, self.H // self.patchsize_H - 1)
        col = random.randint(0, self.W // self.patchsize_W - 1)
        ctx["patch_idx"] = (row, col)
        return jsonfy({"row": row, "col": col})