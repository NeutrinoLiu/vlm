
from templates_lib.func import *
from templates_lib.QA import QATemplate, QAMCTemplate
from templates_lib.task import TaskSet, Hints

class MovementTasks(TaskSet):
    def __init__(self, captioner=None, basefilter=None, cfg={}, seed=0):
        super().__init__(cfg, seed)

        obj_desc = obj_desc_fn if captioner is None else captioner.obj_desc_fn
        roi_frame_only = cfg.get("roi_frame_only", False)
        myfilter = basefilter if basefilter is not None else lambda x: True
        gen_perturb = lambda : random.uniform(0.1, 0.2)

        # --------------- movement of an object over a range of frames --------------- #
        single_obj_abs_movement = QATemplate(
            Q_temp="What is the movement distance of <obj> between frame <frame_range> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_movement>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("frame_range", frame_range_fn(roi_frame_only)),
                ("abs_movement", obj_movement_fn(0)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_frames": 5,
                "QA_type": "single_obj_abs_movement"
                },
        )

        # ------- movement of an object over a range of frames with MC ------- #
        single_obj_abs_movement_MC = QAMCTemplate(
            Q_temp=f"What is the movement distance of <obj> between frame <frame_range> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("frame_range", frame_range_fn(roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_options": 5,
                "num_frames": 5,
                "opt_mapper_gen": roulette(obj_movement_fn(0), max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "single_obj_abs_movement_MC"
            },
        )

        # ---------- relative movement of two objects over a range of frames --------- #
        multiple_obj_relative_movement = QAMCTemplate(
            Q_temp=f"Among the following objects, which one has the <minmax> movement distance between frame <frame_range> ? <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ("minmax", minmax("shortest", "longest")),
                ("frame_range", frame_range_fn(roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_options": 2,
                "num_frames": 5,
                "opt_mapper_gen": obj_desc,
                "ans_index_gen": index_of_minmax_fn(obj_movement_fn),
                "QA_type": "multiple_obj_relative_movement"
            },
        )

        # --------------- movement of ego camera over a range of frames -------------- #
        ego_movement = QATemplate(
            Q_temp="What is the movement distance of the ego camera between frame <frame_range> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_movement>",
            obj_mappers=[
                ("frame_range", frame_range_fn(roi_frame_only)),
                ("abs_movement", ego_movement_calc),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 0,
                "num_frames": 5,
                "QA_type": "ego_movement"
            },
        )

        # ------- movement of ego camera over a range of frames with MC ------- #
        ego_movement_MC = QAMCTemplate(
            Q_temp=f"What is the movement distance of the ego camera between frame <frame_range> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ("frame_range", frame_range_fn(roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 0,
                "num_options": 5,
                "num_frames": 5,
                "opt_mapper_gen": roulette(ego_movement_calc, max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "ego_movement_MC"
            },
        )

        self.tasks.update({
            "single_obj_abs_movement": single_obj_abs_movement,
            "single_obj_abs_movement_MC": single_obj_abs_movement_MC,
            "multiple_obj_relative_movement": multiple_obj_relative_movement,
            "ego_movement": ego_movement,
            "ego_movement_MC": ego_movement_MC,
        })