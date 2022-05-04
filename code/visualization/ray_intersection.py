import trimesh
import numpy as np

# faces = np.load('/home/chen/IPNet/faces.npy')
# pcl = trimesh.load('/home/chen/Desktop/buff/smpl_pcl.ply')
smpl_mesh = trimesh.load('/home/chen/Desktop/smpl_mesh.obj', process=False) # trimesh.primitives.Sphere()

bbox = smpl_mesh.bounding_box # .apply_scale(1.2) we dont need it for now since we load the scaled mesh already
# create some rays
ray_origins = np.load('/home/chen/Desktop/bbox_sampler/cam_loc.npy')
ray_directions = np.load('/home/chen/Desktop/buff/ray_dirs.npy')

"""
Signature: mesh.ray.intersects_location(ray_origins,
                                        ray_directions,
                                        multiple_hits=True)
Docstring:
Return the location of where a ray hits a surface.
Parameters
----------
ray_origins:    (n,3) float, origins of rays
ray_directions: (n,3) float, direction (vector) of rays
Returns
---------
locations: (n) sequence of (m,3) intersection points
index_ray: (n,) int, list of ray index
index_tri: (n,) int, list of triangle (face) indexes
"""
n_pix = ray_origins.shape[0]
# run the mesh- ray test
locations, index_ray, index_tri = bbox.ray.intersects_location(
    ray_origins=ray_origins,
    ray_directions=ray_directions,
    multiple_hits=True)
import ipdb
ipdb.set_trace()

num_ray_hits = locations.shape[0] // 2

unhit_first_index_ray = set(np.arange(0, 512)).difference(index_ray[:num_ray_hits])
unhit_second_index_ray = set(np.arange(0, 512)).difference(index_ray[num_ray_hits:])
# assert unhit_first_index_ray == unhit_second_index_ray, "unhitted first index ray and second index ray are not the same"
unhit_index_ray = list(unhit_first_index_ray)
to_pad_index_ray = np.random.choice(index_ray[:num_ray_hits].shape[0], len(unhit_index_ray))

near_hit_locations = np.zeros((n_pix, locations.shape[1]))
near_hit_locations[index_ray[:num_ray_hits]] = locations[:num_ray_hits]
# padding the invalid two hits 
near_hit_locations[unhit_index_ray] = locations[:num_ray_hits][to_pad_index_ray]

far_hit_locations = np.zeros((n_pix, locations.shape[1]))
far_hit_locations[index_ray[num_ray_hits:]] = locations[num_ray_hits:]
# padding the invalid two hits
far_hit_locations[unhit_index_ray] = locations[num_ray_hits:][to_pad_index_ray]

near = np.linalg.norm(ray_origins - near_hit_locations, axis=1)

far = np.linalg.norm(ray_origins - far_hit_locations, axis=1)

ray_directions[unhit_index_ray] = ray_directions[to_pad_index_ray]

# stack rays into line segments for visualization as Path3D
ray_visualize = trimesh.load_path(np.hstack((
    ray_origins,
    ray_origins + ray_directions)).reshape(-1, 2, 3))

# make mesh transparent- ish
bbox.visual.face_colors = [100, 100, 100, 100]

# create a visualization scene with rays, hits, and mesh
scene = trimesh.Scene([
    [bbox, smpl_mesh],
    ray_visualize,
    trimesh.points.PointCloud(locations)])

# display the scene
scene.show()