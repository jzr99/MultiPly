import os
import tqdm
import trimesh
import numpy as np
import trimesh
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.renderables.meshes import Meshes
from aitviewer.utils.so3 import aa2rot_numpy
import numpy as np
from scipy.spatial.transform import Rotation as R
if __name__ == '__main__':

    # frames_folder = "/home/chen/disk2/motion_capture/wenting/meshes"

    vertices = []
    faces = []
    vertex_normals = []
    uvs = []
    texture_paths = []

    mesh = trimesh.load(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/outdoors_fencing_01_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled/0337_deformed.ply'), process=False)
    mesh = trimesh.smoothing.filter_humphrey(mesh, iterations=1)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    ours_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='ours', flat_shading=False)

    # mesh = trimesh.load(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Wildmotion1_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled/0028_deformed.ply'), process=False)
    # mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    # ours_mesh2 = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='ours2', flat_shading=True)

    mesh = trimesh.load('/home/chen/disk2/ICON_new_results/outdoors_fencing_01/icon-filter/obj/0337_refine.obj')
    mesh = trimesh.smoothing.filter_humphrey(mesh, iterations=1)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    icon_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='icon', flat_shading=False)

    mesh = trimesh.load('/home/chen/disk2/SelfRecon_results/outdoors_fencing_01/result/final_meshes_transformed/0337.ply')
    mesh = trimesh.smoothing.filter_humphrey(mesh, iterations=1)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    # rotation1 = R.from_rotvec([0, 0, np.pi])
    # rotation2 = R.from_rotvec([0, np.pi, 0])
    # rotation = rotation1 * rotation2
    rotation = R.from_rotvec([0, 0, 0])
    selfrecon_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='selfrecon', flat_shading=False, rotation=rotation.as_matrix())

    # mesh = trimesh.load('/home/chen/ml-neuman/debug_output_posed.obj', process=False)
    # neuman_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='neuman', flat_shading=True)

    viewer = Viewer()
    viewer.scene.add(ours_mesh, selfrecon_mesh, icon_mesh)
    # viewer.scene.add(selfrecon_mesh)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    viewer.run()