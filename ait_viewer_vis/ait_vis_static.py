import os
import tqdm
import trimesh
import numpy as np
import trimesh
from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.renderables.meshes import Meshes
import numpy as np
if __name__ == '__main__':

    # frames_folder = "/home/chen/disk2/motion_capture/wenting/meshes"

    vertices = []
    faces = []
    vertex_normals = []
    uvs = []
    texture_paths = []

    mesh = trimesh.load(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/bike_wo_disp_freeze_20_every_20_opt_pose_split/test_animation_mesh/0000_deformed.ply'), process=False)
    ours_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='ours', flat_shading=True)

    mesh = trimesh.load('/home/chen/ml-neuman/debug_output_posed.obj', process=False)
    neuman_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='neuman', flat_shading=True)

    viewer = Viewer()
    viewer.scene.add(ours_mesh, neuman_mesh)
    # viewer.scene.add(ours_mesh)
    viewer.run()