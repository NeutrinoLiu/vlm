def filter_area(anno, thres=5_000):
    return anno['2d_crop']["area"] > thres

def filter_visiblity(anno, thres=100):
    vis = anno['visibility'].split("-")[-1]
    return int(vis) >= thres

def filter_all(*filters):
    def filter_fn(anno):
        for f in filters:
            if not f(anno):
                return False
        return True
    return filter_fn

def black_list_fn(
        black_list=[
            "movable_object.trafficcone",
            "movable_object.barrier",
        ]):
    def fn(anno):
        if anno['category_name'] in black_list:
            return False
        return True
    return fn