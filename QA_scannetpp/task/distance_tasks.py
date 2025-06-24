
from templates_lib.func import *
from templates_lib.QA import QATemplate, QAMCTemplate
from templates_lib.task import TaskSet, Hints

class DistTasks(TaskSet):
    def __init__(self, captioner=None, basefilter=None, cfg={}, seed=0):
        super().__init__(cfg, seed)

        obj_desc = obj_desc_fn if captioner is None else captioner.obj_desc_fn
        roi_frame_only = cfg.get("roi_frame_only", False)
        myfilter = basefilter if basefilter is not None else lambda x: True
        gen_perturb = lambda : random.uniform(0.1, 0.2)

        # ----------- distance of an object to ego cam at a specific frame ----------- #
        single_obj_abs_dist = QATemplate(
            Q_temp="What is the distance between <obj> and the ego camera at frame <frame> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_dist>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("frame", frame_idx_fn(0, roi_frame_only)),
                ("abs_dist", obj_cam_dist_fn(0)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_frames": 1,
                "QA_type": "single_obj_abs_dist"
                },
        )

        # ------- distance of an object to ego cam at a specific frame with MC ------- #
        single_obj_abs_dist_MC = QAMCTemplate(
            Q_temp=f"What is the distance between <obj> and the ego camera at frame <frame> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("frame", frame_idx_fn(0, roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_options": 5,
                "num_frames": 1,
                "opt_mapper_gen": roulette(obj_cam_dist_fn(0), max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "single_obj_abs_dist_MC"
            },
        )

        # ----------- distance between two objects at a specific frame ----------- #
        double_obj_abs_dist = QATemplate(
            Q_temp="What is the distance between <obj1> and <obj2> at frame <frame> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_dist>",
            obj_mappers=[
                ("obj1", obj_desc(0)),
                ("obj2", obj_desc(1)),
                ("frame", frame_idx_fn(0, roi_frame_only)),
                ("abs_dist", obj_dist_between),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_frames": 1,
                "QA_type": "double_obj_abs_dist"
            },
        )

        # --------- distance between two objects at a specific frame with MC --------- #
        double_obj_abs_dist_MC = QAMCTemplate(
            Q_temp=f"What is the distance between <obj1> and <obj2> at frame <frame> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ("obj1", obj_desc(0)),
                ("obj2", obj_desc(1)),
                ("frame", frame_idx_fn(0, roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_options": 5,
                "num_frames": 1,
                "opt_mapper_gen": roulette(obj_dist_between, max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "double_obj_abs_dist_MC"
            },
        )

        # ----- minmax dist between object and ego camera over a range of frames ----- #
        single_obj_minmax_dist = QATemplate(
            Q_temp="What is the <minmax> distance between <obj> and the ego camera between frame <frame_range> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_dist>",
            obj_mappers=[
                ('minmax', minmax()),
                ("obj", obj_desc(0)),
                ("frame_range", frame_range_fn(roi_frame_only)),
                ("abs_dist", obj_cam_dist_minmax),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_frames": 5,
                "QA_type": "single_obj_minmax_dist"
                },
        )

        # - minmax dist between object and ego camera over a range of frames with MC - #
        single_obj_minmax_dist_MC = QAMCTemplate(
            Q_temp=f"What is the <minmax> distance between <obj> and the ego camera between frame <frame_range> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ('minmax', minmax()),
                ("obj", obj_desc(0)),
                ("frame_range", frame_range_fn(roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_options": 5,
                "num_frames": 5,
                "opt_mapper_gen": roulette(obj_cam_dist_minmax, max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "single_obj_minmax_dist_MC"
            },
        )

        # ----- minmax dist between two objects over a range of frames ----- #
        double_obj_minmax_dist = QATemplate(
            Q_temp="What is the <minmax> distance between <obj1> and <obj2> between frame <frame_range> ? "
            + Hints.DIST_REPLY_HINT,
            A_temp="<abs_dist>",
            obj_mappers=[
                ('minmax', minmax()),
                ("obj1", obj_desc(0)),
                ("obj2", obj_desc(1)),
                ("frame_range", frame_range_fn(roi_frame_only)),
                ("abs_dist", obj_dist_between_minmax),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_frames": 5,
                "QA_type": "double_obj_minmax_dist"
            },
        )

        # - minmax dist between two objects over a range of frames with MC - #
        double_obj_minmax_dist_MC = QAMCTemplate(
            Q_temp=f"What is the <minmax> distance between <obj1> and <obj2> between frame <frame_range> ? choose the closest one among the following options. <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ('minmax', minmax()),
                ("obj1", obj_desc(0)),
                ("obj2", obj_desc(1)),
                ("frame_range", frame_range_fn(roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_options": 5,
                "num_frames": 5,
                "opt_mapper_gen": roulette(obj_dist_between_minmax, max_opts=5, perturb=gen_perturb()),
                "ans_index_gen": index_of_roulette,
                "QA_type": "double_obj_minmax_dist_MC"
            },
        )

        # ----- relative dist between multiple objects at a specific frame ----- #
        multiple_obj_relative_dist = QAMCTemplate(
            Q_temp=f"Among the following objects, which one is the <minmax> the ego camera at frame <frame> ? <{QAMCTemplate.OPT_PREFIX}> "
            + Hints.MULTI_CHOICE_REPLY_HINT,
            A_temp=f"<{QAMCTemplate.ANS_PREFIX}>",
            obj_mappers=[
                ('minmax', minmax("closest to", "farthest from")),
                ("frame", frame_idx_fn(0, roi_frame_only)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 2,
                "num_options": 2,
                "num_frames": 1,
                "opt_mapper_gen": obj_desc,
                "ans_index_gen": index_of_minmax_fn(obj_cam_dist_fn),
                "QA_type": "multiple_obj_relative_dist"
            },
        )

        # ----- local coordinates of an object in ego camera coordinate ----- #
        local_coords = QATemplate(
            Q_temp="What is the local coordinates of <obj> in the ego camera coordinate at frame <frame> ? "
            + Hints.XY_COORD_REPLY_HINT,
            A_temp="<local_coords>",
            obj_mappers=[
                ("obj", obj_desc(0)),
                ("frame", frame_idx_fn(0, roi_frame_only)),
                ("local_coords", obj_local_coords_fn(0)),
            ],
            obj_filter=myfilter,
            config={
                "num_objs": 1,
                "num_frames": 1,
                "QA_type": "local_coords"
                },
        )

        self.tasks.update({
            "single_obj_abs_dist": single_obj_abs_dist,
            "single_obj_abs_dist_MC": single_obj_abs_dist_MC,
            "double_obj_abs_dist": double_obj_abs_dist,
            "double_obj_abs_dist_MC": double_obj_abs_dist_MC,
            "single_obj_minmax_dist": single_obj_minmax_dist,
            "single_obj_minmax_dist_MC": single_obj_minmax_dist_MC,
            "double_obj_minmax_dist": double_obj_minmax_dist,
            "double_obj_minmax_dist_MC": double_obj_minmax_dist_MC,
            "multiple_obj_relative_dist": multiple_obj_relative_dist,
            "local_coords": local_coords
        })