import imp
import os
import json
from re import I
import shutil
from tqdm import tqdm
# from cv2 import transform
# from numpy import imag
from PIL import Image

import sys
sys.path.append('../')



import json
import numpy as np
from scipy.spatial.transform import Rotation as R

from scannetpp.common.utils.colmap import read_model

import numpy as np
from PIL import ImageDraw
# import open3d as o3d
from pathlib import Path
from multiprocessing import Pool


def read_txt_list(path):
    import os;
    with open(path) as f:
        lines = f.read().splitlines()

    return lines

scene_list = read_txt_list("./sc_name.txt")
# scene_list = ["0a5c01343testtest"] 


SCANNETPP_ROOT = "../Journey9ni_raw_data_part/scannetpp/data"
OUTPUT_ROOT = "./structured-data"
CROPS_ROOT = "./structured-data-crops"
POSTFIX = "jpg"
SENSOR = 'CAM_FRONT'
MIN_AREA = 10000
MUILTI_PROCESS_NUM = 8



def load_camera_params_as_dict(colmap_dir, json_path):

    cameras, images, _ = read_model(colmap_dir, ext='.txt')

    cam_dict = {}
    for image_id, image in images.items():
        file_name = image.name.split("/")[-1]

        camera = cameras[image.camera_id]
        model = camera.model
        params = camera.params

        if model in ['PINHOLE', 'OPENCV', 'OPENCV_FISHEYE']:
            fx, fy, cx, cy = params[:4]
            intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]
            ])
        else:
            raise NotImplementedError(f"Unsupported camera model: {model}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)

            # 构造 intrinsic matrix
            fl_x = data['fl_x']
            fl_y = data['fl_y']
            cx = data['cx']
            cy = data['cy']
            intrinsic = np.array([
                [fl_x,   0.0,  cx],
                [0.0,    fl_y, cy],
                [0.0,    0.0,  1.0]
            ])

        world_to_cam = image.world_to_camera
        cam_to_world = np.linalg.inv(world_to_cam)

        translation = cam_to_world[:3, 3]
        rotation_matrix = cam_to_world[:3, :3]
        quaternion = R.from_matrix(rotation_matrix).as_quat()

        qx, qy, qz, qw = R.from_matrix(rotation_matrix).as_quat()
        quaternion = [qw, qx, qy, qz]  # 变成 scalar-first 顺序


        cam_dict[file_name] = {
            'cam_t': translation.tolist(),
            'cam_r': quaternion,
            'intrinsic': intrinsic,
            'transform_matrix': cam_to_world.tolist()
        }

    return cam_dict

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


# for sc in scene_list:
def process_single_scene(sc):
    from scannetpp.common.utils.anno import get_visiblity_from_cache
    from scannetpp.common.scene_release import ScannetppScene_Release
    with open(os.path.join(SCANNETPP_ROOT, sc, 'scans', 'segments_anno.json'), 'r') as f:
        data = json.load(f)
    inst_crops = {} # token: [image_path, 2d_crop_diag, 2d_crop_area, visb]
    visibility = get_visiblity_from_cache(scene = ScannetppScene_Release(sc, data_root=SCANNETPP_ROOT),
                                          raster_dir=Path("../temp_rasterize/dslr"),
                                          cache_dir="../temp_cache",
                                          image_type="dslr",
                                          subsample_factor=1,)

    # print(f" > Processing scene {sc['token']} ...")
    sc_dir = os.path.join(OUTPUT_ROOT, sc)
    frame_list = sorted(os.listdir(os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'undistorted_images')))
    # frame_list = sorted(os.listdir(os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'undistorted_images')))
    
    colmap_dir = os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'colmap')
    json_path = os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'nerfstudio','transforms_undistorted.json')
    camera_map = load_camera_params_as_dict(colmap_dir, json_path)

    # 只保留 train split 中的帧
    train_test_path = os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'train_test_lists.json')
    with open(train_test_path, 'r') as f:
        train_frames = set(json.load(f)['train'])

    # 过滤掉不在 train split 中的图像
    frame_list = [f for f in frame_list if f in train_frames]

    a=0

    for frame in tqdm(frame_list, desc=f"Processing scene {sc}"):

        frame_dir = os.path.join(sc_dir, str(frame[:-4]))  
        os.makedirs(frame_dir, exist_ok=True)
        raw_path = os.path.join(SCANNETPP_ROOT, sc, 'dslr', 'undistorted_images', frame)

        rgb_path = os.path.join(frame_dir, f'{SENSOR}_raw.{POSTFIX}')
        rgb_box_path = os.path.join(frame_dir, f'{SENSOR}_box.{POSTFIX}')
        shutil.copy(raw_path, rgb_path)


        frame_data = camera_map[frame]

        image_width, image_height = Image.open(rgb_path).size

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
            'intrinsic': frame_data['intrinsic'].tolist(),  # Convert to list for JSON serialization
            'annos' : [],
        }


        a=0
        annos = []
        img = Image.open(rgb_path)
        vis_objects = []
        for obj in data['segGroups']:
                # 跳过在该frame中不可见的物体
            obj_id = obj['objectId']

            if obj_id not in visibility["images"][frame]["objects"]:
                continue  # 当前帧中没这个 object，跳过

            vis_info = visibility["images"][frame]["objects"][obj_id]

            if vis_info["visible_vertices_frac"] < 0.1:
                continue

            centroid = obj['obb']['centroid']
            axes_lengths = obj['obb']['axesLengths']
            normalized_axes = obj['obb']['normalizedAxes']

            corners_3d = obb_to_corners(centroid, axes_lengths, normalized_axes)
            corners_cam = world_to_camera(corners_3d, frame_data['transform_matrix'])


            if np.any(corners_cam[2, :] <= 0):
                continue

            corners_2d = project_to_image(corners_cam, frame_data['intrinsic'])
            x_min, y_min = np.min(corners_2d, axis=1)
            x_max, y_max = np.max(corners_2d, axis=1)

            # if x_max < 0 or y_max < 0 or x_min > image_width or y_min > image_height:
            #     continue

            # 新逻辑（所有 corner 都必须在图内）
            inside_mask = (
                (corners_2d[0, :] >= 0) & (corners_2d[0, :] < image_width) &
                (corners_2d[1, :] >= 0) & (corners_2d[1, :] < image_height)
            )

            if not np.all(inside_mask):
                continue  # 有一个角点不在图内，跳过

            # if not np.any(inside_mask):
            #     continue  # 全部角点都不在图内，跳过


            # 画框（不保存）
            img = draw_projected_box(img, corners_2d)
            
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

# scene_list = scene_list[:12]
with Pool(processes=MUILTI_PROCESS_NUM) as pool:  # 你可设置为 os.cpu_count()
    for _ in tqdm(pool.imap_unordered(safe_process, scene_list), total=len(scene_list)):
        pass
