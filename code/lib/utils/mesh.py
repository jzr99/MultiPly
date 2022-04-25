import numpy as np
import torch
from skimage import measure
from ..libmise import mise
import trimesh


def mesh_from_implicit_func(func,
                            bbox,
                            resolution=(256, 256, 256),
                            level_set=0,
                            extractt_max_component=False,
                            coarse_bbox=False):
    resolution = np.asarray(resolution)
    if coarse_bbox:
        grid, scale, offset = create_grid(bbox, resolution // 2)
        val = MISE(func, grid)
        verts, faces, _, _ = measure.marching_cubes(
            val.transpose(1, 0, 2), level_set, gradient_direction="descent")
        verts = verts / (resolution // 2) - 0.5
        verts = verts * scale + offset

        max_verts = verts.max(axis=0)
        min_verts = verts.min(axis=0)
        scale = max_verts - min_verts

        bbox = (min_verts - 0.2 * scale, max_verts + 0.2 * scale)

    grid, scale, offset = create_grid(bbox, resolution)
    val = MISE(func, grid)
    verts, faces, _, _ = measure.marching_cubes(
        val.transpose(1, 0, 2), level_set, gradient_direction="descent")
    verts = verts / resolution - 0.5
    verts = verts * scale + offset

    mesh = trimesh.Trimesh(verts, faces)
    if extractt_max_component:
        # remove disconnect part
        connected_comp = mesh.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        mesh = max_comp
        return mesh
    else:
        return mesh


def create_grid(bbox, resolution):
    # create voxel grid
    resX, resY, resZ = resolution
    grid = np.stack(np.meshgrid(np.linspace(-0.5, 0.5, resX),
                                np.linspace(-0.5, 0.5, resY),
                                np.linspace(-0.5, 0.5, resZ)),
                    axis=0)

    # scale according to bbox
    bbox_min, bbox_max = bbox
    scale = (bbox_max - bbox_min)
    offset = (bbox_max + bbox_min) / 2

    grid = grid.reshape(3, -1)
    grid = grid * scale[:, np.newaxis] + offset[:, np.newaxis]
    return grid.reshape(3, *resolution), scale, offset


def batch_eval(eval_func, pts, max_size=8192):
    outputs = []
    for batch in np.array_split(pts, (len(pts) + max_size - 1) // max_size):
        # import pdb
        # pdb.set_trace()
        outputs.append(eval_func(batch))
    return np.concatenate(outputs, axis=0)


def MISE(eval_fn, grid, threshold=0.0005, step=1):
    """Numpy implementation for Multiresolution IsoSurface Extraction

    The idea is to evalaute voxel grid from coarse to fine, and skip the subvoxel
        if the difference between max and min value is below the required threshold.

    Args:
        eval_fn: func(points) -> values (sdf / occupancy / etc.)
        grid (3, resX, resY, resZ): 3D grid voxels
        threshold (float): threshold for whether further division is required
        step (int): size of initial subvoxel

    Return:
        (resX, resY, resZ)
    """
    resolution = grid.shape[1:]
    values = np.zeros(resolution, dtype=float)
    to_eval = np.zeros(resolution, dtype=bool)
    unoccupied = np.ones(resolution, dtype=bool)
    while step > 0:
        # TODO: replace to_eval with indices?
        to_eval[::step, ::step, ::step] = True
        mask = np.logical_and(to_eval, unoccupied)
        # import pdb
        # pdb.set_trace()
        values[mask] = batch_eval(eval_fn, grid[:, mask].T)
        if step <= 1:
            # has already reached leaf nodes
            break
        resX, resY, resZ = resolution
        indices = np.mgrid[:resX-step:step, :resY-step:step, :resZ-step:step]
        indices = indices.reshape(3, -1)
        for index in indices.transpose():
            if not unoccupied[tuple(index + step // 2)]:
                # if any point within the box has already been occupied
                #   so does every other point (because we are filling box
                #   from coarse to fine)
                continue
            corners = np.mgrid[:2, :2, :2].reshape(3, -1) * step
            vals = values[tuple(index[:, np.newaxis] + corners)]
            diff = vals.max() - vals.min()
            if diff < threshold:
                x, y, z = index
                values[x+1:x+step, y+1:y+step, z+1:z+step] = vals.mean()
                unoccupied[x:x+step, y:y+step, z:z+step] = False
        step = step // 2
    return values

def generate_mesh(func, verts, level_set=0, res_init=32, res_up=3, point_batch=5000):
    
    scale = 1.1  # Scale of the padded bbox regarding the tight one.
    verts = verts.data.cpu().numpy()
    
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    mesh_extractor = mise.MISE(res_init, res_up, level_set)
    points = mesh_extractor.query()
    
    # query occupancy grid
    while points.shape[0] != 0:
        
        orig_points = points
        points = points.astype(np.float32)
        points = (points / mesh_extractor.resolution - 0.5) * scale
        points = points * gt_scale + gt_center
        points = torch.tensor(points).float().cuda()
        
        values = []
        for _, pnts in enumerate((torch.split(points,point_batch,dim=0))):
            out = func(pnts)
            values.append(out['occ'].data.cpu().numpy())
        values = np.concatenate(values, axis=0).astype(np.float64)[:,0]
        
        mesh_extractor.update(orig_points, values)
        
        points = mesh_extractor.query()
    
    value_grid = mesh_extractor.to_dense()
    # value_grid = np.pad(value_grid, 1, "constant", constant_values=-1e6)

    # marching cube
    verts, faces, normals, values = measure.marching_cubes_lewiner(
                                                volume=value_grid,
                                                gradient_direction='ascent',
                                                level=level_set)

    verts = (verts / mesh_extractor.resolution - 0.5) * scale
    verts = verts * gt_scale + gt_center
    faces = faces[:, [0,2,1]]
    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    #remove disconnect part
    connected_comp = meshexport.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport = max_comp

    return meshexport


