import os
import numpy as np
import os
import torch
from pathlib import Path
from typing import NamedTuple, Optional
from PIL import Image
import trimesh
import cv2  # Assuming OpenCV is used for image saving
from scene.gaussian_model import BasicPointCloud
from scene.dataset_readers import storePly
from scene.colmap_loader import rotmat2qvec
from utils.graphics_utils import focal2fov, fov2focal
from dust3r.utils.device import to_numpy


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Optional[np.ndarray] = None
    mono_depth: Optional[np.ndarray] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None

    
def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)    
    sparse_path.mkdir(exist_ok=True, parents=True)
    
    return save_path, images_path, masks_path, sparse_path


def save_images_masks(imgs, masks, images_path, masks_path, mask_images=False):
    # Saving images and optionally masks/depth maps
    for i, (image, mask) in enumerate(zip(imgs, masks)):
        image_save_path = images_path / f"{i}.png"
        
        mask_save_path = masks_path / f"{i}.png"
        if mask_images:
            image[~mask] = 1.
        rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)
        
        
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    # Save cameras.txt
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")

        
def save_imagestxt(world2cam, sparse_path):
     # Save images.txt
    images_file = sparse_path / 'images.txt'
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            # Convert rotation matrix to quaternion
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here


def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                int(color[0]), int(color[1]), int(color[2]),
                normal[0], normal[1], normal[2]
            ))
            

def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct#, pts


def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = sparse_path / 'points3D.ply'
    pc = get_pc(imgs, pts3d, msk)
    
    pc.export(save_path)
    

def normalize_scene(
    pts_list,
    masks,
    c2w_list,
):
    """
    Normalizes a list of point clouds and adjusts camera parameters accordingly.

    Parameters:
    - pts_list: list of torch.Tensor of shape (res, res, 3), per-image point clouds.
    - masks: np.array of shape (len(pts_list), res, res), dtype=bool, per-image masks.
    - c2w_list: list of np.array of shape (4, 4), per-image camera-to-world matrices.

    Returns:
    - pts_normalized_list: list of torch.Tensor of shape (res, res, 3), normalized point clouds.
    - c2w_normalized_list: list of np.array of shape (4, 4), adjusted camera-to-world matrices.
    """
    # **Step 1: Collect all valid points from all images**
    valid_pts_all = []

    for idx, pts_i in enumerate(pts_list):
        # Get the corresponding mask from masks array
        mask_i = masks[idx]

        # Convert mask to torch.Tensor and flatten pts_i and mask_i
        mask_i_tensor = torch.from_numpy(mask_i.astype(bool)).to(pts_i.device)
        pts_i_flat = pts_i.view(-1, 3)
        mask_i_flat = mask_i_tensor.view(-1)

        # Get valid points where mask is True
        valid_pts_i = pts_i_flat[mask_i_flat]
        valid_pts_all.append(valid_pts_i)

    # Concatenate all valid points from all images
    valid_pts_all = torch.cat(valid_pts_all, dim=0)

    # **Step 2: Compute centroid and scaling factor using all valid points**
    centroid = valid_pts_all.mean(dim=0)
    scale = valid_pts_all.sub(centroid).norm(dim=1).max()

    # Convert centroid and scale to NumPy for consistent use with c2w
    centroid_cpu = centroid.cpu().numpy()
    scale_cpu = scale.cpu().item()

    # **Step 3: Normalize the points and adjust camera parameters**
    pts_normalized_list = []
    c2w_normalized_list = []

    for idx, pts_i in enumerate(pts_list):
        c2w_i = c2w_list[idx]

        # Normalize the point cloud
        pts_i_normalized = (pts_i - centroid) / scale
        pts_normalized_list.append(pts_i_normalized)

        # Adjust the camera-to-world matrix
        c2w_translation = c2w_i[:3, 3]
        c2w_translation_normalized = (c2w_translation - centroid_cpu) / scale_cpu
        c2w_i_normalized = c2w_i.copy()
        c2w_i_normalized[:3, 3] = c2w_translation_normalized
        c2w_normalized_list.append(c2w_i_normalized)


    return pts_normalized_list, c2w_normalized_list
