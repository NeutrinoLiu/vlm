import random

class Hints:
    DIST_UNIT_HINT = "unit in meters"
    FRAME_IDX_HINT = "frame idx starts from 0"
    DIST_REPLY_HINT = f"(just reply in json {{'dist': dist}}, {DIST_UNIT_HINT}, {FRAME_IDX_HINT})"
    MULTI_CHOICE_REPLY_HINT = f"(just reply the correct option's letter in json {{'ans': ans}}, {DIST_UNIT_HINT}, {FRAME_IDX_HINT})"
    XY_COORD_REPLY_HINT = f"(just reply in json {{'x': x, 'y': y}}, x to the right, y to the front, ignore z, {DIST_UNIT_HINT}, {FRAME_IDX_HINT})"

class TaskSet:
    def __init__(self, cfg, seed):
        self.tasks = {}
        self.cfg = cfg
        self.randseed = seed
    def produce(self, dataset, num_qas, verbose=False):
        qas = []
        stats = {}
        if isinstance(num_qas, int):
            while len(qas) < num_qas:
                sc = random.choice(dataset.scenes)
                temp = random.choice(list(self.tasks.values()))
                # qa = temp(sc)
                try:
                    qa = temp(sc)
                except Exception as e:
                    if verbose:
                        print(f"Error: {e}")
                    continue
                qas.append(qa)
                stats[qa.QA_type] = stats.get(qa.QA_type, 0) + 1
                if len(qas) % 500 == 0:
                    print(f"Generated {len(qas)} QAs, stats: {stats}")
        elif isinstance(num_qas, dict):
            for k, v in num_qas.items():
                qas_one_type = []
                if k not in self.tasks:
                    if verbose:
                        print(f"Warning: {k} not found in tasks")
                    continue
                while len(qas_one_type) < v:
                    sc = random.choice(dataset.scenes)
                    temp = self.tasks[k]
                    try:
                        qa = temp(sc)
                    except Exception as e:
                        if verbose:
                            print(f"Error: {e}")
                        continue
                    qas_one_type.append(qa)
                    stats[k] = stats.get(k, 0) + 1
                qas.extend(qas_one_type)
                print(f"Generated {len(qas_one_type)} QAs of type {k}")
        
        stats = dict(sorted(stats.items()))
        return qas, stats


class MultiTaskSet(TaskSet):
    def __init__(self, subsets: list, cfg={}, seed=0, **kwargs):
        super().__init__(cfg, seed)
        for task_cls in subsets:
            assert issubclass(task_cls, TaskSet), f"Task {task_cls} is not a subclass of TaskSet"
            task_cls = task_cls(**kwargs, cfg=cfg, seed=seed)
            self.tasks.update(task_cls.tasks)