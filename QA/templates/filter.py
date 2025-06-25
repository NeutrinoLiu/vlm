def filter_area(anno, thres=5_000):
    return anno['2d_crop']["area"] > thres

def filter_area_fn(min_area, max_area=None):
    def filter_fn(anno):
        area = anno['2d_crop']["area"]
        if max_area is None:
            return area > min_area
        else:
            return min_area <= area <= max_area
    return filter_fn

def filter_visiblity(anno, thres=100):
    try:
        vis = anno['visibility'].split("-")[-1]
    except:
        vis = thres
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
        for b in black_list:
            if b in anno['category_name']:
                return False
        return True
    return fn