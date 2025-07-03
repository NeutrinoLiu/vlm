import imp
import os
import json
import cv2

import shutil
from tqdm import tqdm
# from cv2 import transform
# from numpy import imag
from PIL import Image

import numpy as np
from scipy.spatial import ConvexHull
import sys
sys.path.append('../')



import json
from scipy.spatial.transform import Rotation as R

from scannetpp.common.utils.colmap import read_model

from PIL import ImageDraw
# import open3d as o3d
from pathlib import Path
from multiprocessing import Pool
from shapely.geometry import Polygon, box

def read_txt_list(path):
    import os;
    with open(path) as f:
        lines = f.read().splitlines()

    return lines

scene_list = read_txt_list("./sc_name_arkit.txt")
# scene_list = ["0a5c01343testtest"] 


ARKit_ROOT = "../ARKitScenes/3dod/Training"
OUTPUT_ROOT = "./structured-data"
CROPS_ROOT = "./structured-data-crops"
POSTFIX = "jpg"
SENSOR = 'CAM_FRONT'
MIN_AREA = 10000
MUILTI_PROCESS_NUM = 8




def parse_traj_to_dict(traj_path):
    """
    Parse a .traj file into a dict mapping rounded timestamp string → pose.

    Returns:
        dict[str → (rx, ry, rz, tx, ty, tz)]
    """
    traj_dict = {}
    with open(traj_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 7:
                continue  # or raise error
            timestamp, rx, ry, rz, tx, ty, tz = vals
            ts_str = f"{timestamp:.3f}"
            traj_dict[ts_str[:-1]] = (rx, ry, rz, tx, ty, tz)
    return traj_dict


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)

def query_nearest(data, query_val, tol=0.02):
    query_val = float(query_val)
    nearest_key = min(data.keys(), key=lambda k: abs(float(k) - query_val))
    if abs(float(nearest_key) - query_val) > tol:
        return None
    return data[nearest_key]


def build_camera_from_frame(traj_dict, pincam_root, frame, image_size=None):
    """
    Build camera info for a single frame using pre-parsed traj_dict.

    Args:
        traj_dict (dict): timestamp (str) → pose tuple
        frame (str): like '47895593_1161475.723'

    Returns:
        dict or None: camera dict or None if data missing
    """
    ts_key = frame.split('_')[-1][:-1]  # '1161475.723'
    fname = f"{frame}.pincam"
    fpath = os.path.join(pincam_root, fname)

    # if ts_key not in traj_dict or not os.path.exists(fpath):
    if not os.path.exists(fpath):

        raise FileNotFoundError(
            f"Missing data for frame {frame}: "
            f"traj_dict key '{ts_key}' or file '{fpath}' not found."
        )
        return None
    
    rx, ry, rz, tx, ty, tz  = query_nearest(traj_dict, ts_key)
    # rx, ry, rz, tx, ty, tz = traj_dict[nearest_float_key]

    with open(fpath, 'r') as f_intr:
        numbers = list(map(float, f_intr.read().strip().split()))
        if len(numbers) != 6:
            raise ValueError(f"{fpath} should have 6 numbers")
        w, h, fx, fy, cx, cy = numbers[:6]
                # 加这块就行了，其它你都别动
        if image_size is not None:
            image_width, image_height = image_size
            fx *= image_width / w
            fy *= image_height / h
            cx *= image_width / w
            cy *= image_height / h

        intrinsic = np.array([
            [fx, 0, cx],
            [0,  fy, cy],
            [0,   0,  1]
        ])

    rot = R.from_rotvec([rx, ry, rz])
    w2c = np.eye(4)
    w2c[:3, :3] = rot.as_matrix()
    w2c[:3, 3] = [tx, ty, tz]
    c2w = np.linalg.inv(w2c)

    qx, qy, qz, qw = R.from_matrix(c2w[:3, :3]).as_quat()
    return {
        'cam_t': c2w[:3, 3].tolist(),
        'cam_r': [qw, qx, qy, qz],
        'intrinsic': intrinsic.tolist(),
        'transform_matrix': c2w.tolist()
    }


def obb_to_corners(centroid, axes_lengths, normalized_axes):
    """
    Convert oriented bounding box (OBB) to 8 corner points in world coordinate.
    :param centroid: [x, y, z]
    :param axes_lengths: [l_x, l_y, l_z]
    :param normalized_axes: 3x3 axis matrix (rows or cols are orthogonal unit vectors)
    :return: numpy array of shape (3, 8) — 8 corners in world coordinate
    """
    if len(normalized_axes) == 9:
        normalized_axes = np.array(normalized_axes).reshape(3, 3)
    else:
        normalized_axes = np.array(normalized_axes)
    centroid = np.array(centroid)
    axes_lengths = np.array(axes_lengths)
    R_mat = np.array(normalized_axes).T  # Transpose: columns are axes

    offsets = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1]
    ]) * 0.5

    corners = [centroid + R_mat @ (axes_lengths * offset) for offset in offsets]
    return np.array(corners).T  # (3, 8)

def world_to_camera(P_world, T_cam_to_world):
    """
    Convert 3D points from world to camera frame.
    :param P_world: (3, N)
    :param T_cam_to_world: (4, 4)
    :return: (3, N)
    """
    T_w2c = np.linalg.inv(T_cam_to_world)
    P_world_homo = np.vstack([P_world, np.ones((1, P_world.shape[1]))])  # (4, N)
    P_cam = T_w2c @ P_world_homo
    return P_cam[:3] 

def project_to_image(P_cam, K):
    """
    Project 3D camera coordinates to 2D image pixel coordinates.
    :param P_cam: (3, N) points in camera frame
    :param K: (3, 3) intrinsic matrix
    :return: (2, N) pixel coordinates
    """
    p = K @ P_cam
    p[:2] /= p[2:]
    return p[:2]

def draw_projected_box(image, corners_2d):
    """
    Draw 3D box projected onto image.
    :param image: PIL.Image
    :param corners_2d: (2, 8) 8 corners in image space
    :return: updated PIL.Image with lines drawn
    """
    draw = ImageDraw.Draw(image)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # front face
        (4, 5), (5, 7), (7, 6), (6, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # sides
    ]

    for i, j in edges:
        p1 = tuple(corners_2d[:, i])
        p2 = tuple(corners_2d[:, j])
        draw.line([p1, p2], fill=(255, 0, 0), width=2)

    return image



def compute_box_3d(scale, transform, rotation):
    scales = [i / 2 for i in scale]
    l, h, w = scales
    center = np.reshape(transform, (-1, 3))
    center = center.reshape(3)
    x_corners = [l, l, -l, -l, l, l, -l, -l]
    y_corners = [h, -h, -h, h, h, -h, -h, h]
    z_corners = [w, w, w, w, -w, -w, -w, -w]
    corners_3d = np.dot(np.transpose(rotation),
                        np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] += center[0]
    corners_3d[1, :] += center[1]
    corners_3d[2, :] += center[2]
    bbox3d_raw = np.transpose(corners_3d)
    return bbox3d_raw

def bboxes(annotation):
    bbox_list = []
    objects = []
    for label_info in annotation["data"]:
        rotation = np.array(label_info["segments"]["obbAligned"]["normalizedAxes"]).reshape(3, 3)
        transform = np.array(label_info["segments"]["obbAligned"]["centroid"]).reshape(-1, 3)
        scale = np.array(label_info["segments"]["obbAligned"]["axesLengths"]).reshape(-1, 3)
        box3d = compute_box_3d(scale.reshape(3).tolist(), transform, rotation)


        box3d = box3d.T  # (3, 8)
        reorder_idx = [0, 4, 1, 5, 3, 7, 2, 6]# 把新逻辑的列顺序重排，匹配 obb_to_corners()
        box3d = box3d[:, reorder_idx]

        obj = {
            "bbox": box3d,  # shape (3, 8)
            "uid": label_info["uid"],
            "label": label_info["label"],
            "attributes": label_info["attributes"],
            "incompleteness": label_info["attributes"]["attributes"]["incompleteness"],
            "occlusion": label_info["attributes"]["attributes"]["occlusion"],
            "normalizedAxes": label_info["segments"]["obbAligned"]["normalizedAxes"],
            "axesLengths": label_info["segments"]["obbAligned"]["axesLengths"],
            "centroid": label_info["segments"]["obbAligned"]["centroid"],
        }

        # obj = {
        #     "bbox": box3d,  # shape (3, 8)
        #     "uid": label_info.get("uid", None),
        #     "label": label_info.get("label", None),
        #     "arrtibutes": label_info.get("attributes", None),
        # }
        objects.append(obj)
    return objects




def compute_projected_area(xy8):
    """
    输入: xy8 是 shape (2, 8) 的 numpy 数组，代表 8 个点的图像平面投影
    输出: 该投影在图像上的覆盖面积（单位为像素面积）
    """
    # 转置为 shape (8, 2)
    pts = xy8.T

    # 计算凸包
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    # 使用 Shoelace formula 计算面积
    x = hull_pts[:, 0]
    y = hull_pts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area

def compute_visible_area(xy8, width, height):
    """
    计算 3D box 投影到图像上的区域，有多少面积在图像内部。

    参数：
        xy8: numpy array, shape (2, 8)，投影到图像上的8个角点
        width, height: 图像宽高（像素）

    返回：
        可见面积（像素数）
    """
    # 转置为 (8, 2) 点列表
    pts = xy8.T

    # Step 1: 获取凸包多边形（投影区域）
    hull = ConvexHull(pts)
    projected_poly = Polygon(pts[hull.vertices])

    # Step 2: 构造图像边界框（[0,0] ~ [width,height]）
    image_box = box(0, 0, width, height)

    # Step 3: 求交集并计算面积
    visible_poly = projected_poly.intersection(image_box)
    visible_area = visible_poly.area

    return visible_area



# for sc in scene_list:
def process_single_scene(sc):

    with open(os.path.join(ARKit_ROOT, sc, sc+'_3dod_annotation.json'), 'r') as f:
        data = json.load(f)
    objects = bboxes(data)
    inst_crops = {} # token: [image_path, 2d_crop_diag, 2d_crop_area, visb]

    # # print(f" > Processing scene {sc['token']} ...")
    sc_dir = os.path.join(OUTPUT_ROOT, sc)
    frame_list = sorted(os.listdir(os.path.join(ARKit_ROOT, sc, sc+'_frames', 'lowres_wide')))

    traj_dict = parse_traj_to_dict(os.path.join(ARKit_ROOT, sc, sc+'_frames', 'lowres_wide.traj'))

    

    for frame in tqdm(frame_list, desc=f"Processing scene {sc}"):
        print(f" > Processing frame {frame} in scene {sc} ...")
        frame_dir = os.path.join(sc_dir, str(frame[:-4]))  


        os.makedirs(frame_dir, exist_ok=True)
        raw_path = os.path.join(ARKit_ROOT, sc,  sc+'_frames', "lowres_wide", frame)

        rgb_path = os.path.join(frame_dir, f'{SENSOR}_raw.{POSTFIX}')
        rgb_box_path = os.path.join(frame_dir, f'{SENSOR}_box.{POSTFIX}')
        shutil.copy(raw_path, rgb_path)



        image_width, image_height = Image.open(rgb_path).size


        camera_map = build_camera_from_frame(
            traj_dict,
            os.path.join(ARKit_ROOT, sc, sc+'_frames', 'lowres_wide_intrinsics'),
            # os.path.join(ARKit_ROOT, sc, sc+'_frames', "lowres_wide.traj"),
            frame[:-4],
            image_size=(image_width, image_height)
        )



        frame_data = camera_map


        

        anno_path = os.path.join(frame_dir, f'{SENSOR}_meta.json')
        meta = {
            'scene_token': sc,
            'sample_token': '_'.join([sc, SENSOR]),
            'sample_data_token': '_'.join([sc, SENSOR, frame[:-4]]),
            'timestamp': frame[:-4],
            'image_path': rgb_path,
            'image_box_path': rgb_box_path,
            'image_width': image_width,
            'image_height': image_height,
            'cam_t': frame_data['cam_t'],
            'cam_r': frame_data['cam_r'],
            'intrinsic': frame_data['intrinsic'],  # Convert to list for JSON serialization
            'annos' : [],
        }


        a=0
        annos = []
        img = Image.open(rgb_path)
        vis_objects = []
        for obj in objects:
            obj_id = obj['uid']

            corners_3d = obj['bbox']  # shape (3, 8)
            corners_cam = world_to_camera(corners_3d, frame_data['transform_matrix'])


            if np.any(corners_cam[2, :] <= 0):
                continue

            corners_2d = project_to_image(corners_cam, frame_data['intrinsic'])
            x_min, y_min = np.min(corners_2d, axis=1)
            x_max, y_max = np.max(corners_2d, axis=1)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_width, x_max)
            y_max = min(image_height, y_max)

            inside_mask = (
                (corners_2d[0, :] >= 0) & (corners_2d[0, :] < image_width) &
                (corners_2d[1, :] >= 0) & (corners_2d[1, :] < image_height)
            )

            # if not np.all(inside_mask):
            #     continue  # 有一个角点不在图内，跳过

            # if not np.any(inside_mask):
            #     continue  # 全部角点都不在图内，跳过

            num_inside = np.count_nonzero(inside_mask)

            if num_inside <= 0:
                continue

            # # print obj attr
            # print(obj["arrtibutes"])

            # if obj["arrtibutes"]['incompleteness'] != '0~20%':
            #     continue
            # 画框（不保存）
            

            corners_2d_area = compute_projected_area(corners_2d)
            visible_area = compute_visible_area(corners_2d, image_width, image_height)
            vis_frac = visible_area / corners_2d_area if corners_2d_area > 0 else 0
            
            
            if vis_frac < 0.7:
                # print(f" > Warning: Object {obj['label']} in frame {frame} has visibility fraction {vis_frac:.2f}. Skipping.")
                continue
            img = draw_projected_box(img, corners_2d)
            # img.save(rgb_box_path)
            # print(f" > Drawn box for object {obj['label']} in frame {frame}.")
            print(obj["label"], "\t", "incompleteness",obj["incompleteness"], "\t", "occlusion", obj["occlusion"], rgb_box_path)

            normalized_axes = obj['normalizedAxes']
            axes_lengths = obj['axesLengths']
            centroid = obj['centroid']
            # contiue
            rotation_matrix = np.array(normalized_axes).reshape(3, 3).T
            rotation = R.from_matrix(rotation_matrix).as_quat().tolist()
                    
            translation = centroid  # 已是 list 或 [x, y, z] 格式

            anno_token = f"{sc}_{SENSOR}_{frame[:-4]}_{obj_id}"
            instance_token = f"{sc}_{SENSOR}_{obj_id}"

            diag = [[x_min, x_max], [y_min, y_max]]
            area = (x_max - x_min) * (y_max - y_min)

            annos.append({
                # 'objectId': obj['objectId'],
                
                'anno_token': anno_token,
                'instance_token': instance_token,
                'category_name': obj['label'],

                'box_t': translation,
                'box_r': rotation,
                'bbox_size': axes_lengths,
                # "visibility": vis_info["visible_pixels_frac"].tolist(),  # 可见像素比例
                "visibility":"v80-100",  # 可见顶点比例



                'attribute': None,

                '2d_crop': {
                    'diag': diag,
                    'area': area,
                }
            })
            vis_objects.append(obj)

            # dump instance crops
            MIN_AREA = 10000
            MIN_VIS = 100
            # vis_val = int(visibility.split('-')[-1])
            crop = inst_crops.get(instance_token, None)
            if crop is None or area > crop[2]:
                if area > MIN_AREA:
                    inst_crops[instance_token] = [rgb_path, diag, area, obj['label']]
        
        meta['annos'] = annos
        with open(anno_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # 所有 box 绘制完再统一保存
        img.save(rgb_box_path)
        # print(f" > Processed frame {frame} in scene {sc}, found {len(annos)} objects.")
        a=0
        # dump instance crops
    crop_sc_dir = os.path.join(CROPS_ROOT, sc)
    os.makedirs(crop_sc_dir, exist_ok=True)
    for inst_token, crop in inst_crops.items():
        img_path, diag, area, label = crop
        diag=np.array(diag)
        img_save_path = os.path.join(crop_sc_dir, f'{inst_token}.{POSTFIX}')
        img_save_path = os.path.join(crop_sc_dir, f'{str(inst_token).zfill(4)}_{label}.{POSTFIX}')

        img = Image.open(img_path)
        img = img.crop((diag[0, 0], diag[1, 0], diag[0, 1], diag[1, 1]))
        img.save(img_save_path)
            
a=0

def safe_process(sc):
    try:
        return process_single_scene(sc)
    except Exception as e:
        import traceback
        print(f"[ERROR] Scene {sc} failed with error: {e}")
        traceback.print_exc()
        return None

scene_list = [sc.strip() for sc in scene_list if sc.strip()]
# scene_list = scene_list[:12]
with Pool(processes=MUILTI_PROCESS_NUM) as pool:  # 你可设置为 os.cpu_count()
    for _ in tqdm(pool.imap_unordered(safe_process, scene_list), total=len(scene_list)):
        pass
