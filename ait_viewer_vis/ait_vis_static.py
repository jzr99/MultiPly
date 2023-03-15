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


    mesh_256 = trimesh.load(os.path.join('/home/chen/Vid2Avatar_release/code/outputs/ThreeDPW/seattle_wo_disp_freeze_20_every_20_opt_pose/test_mesh_256/0020_deformed.ply'), process=False)
    mesh_512 = trimesh.load(os.path.join('/home/chen/Vid2Avatar_release/code/outputs/ThreeDPW/seattle_wo_disp_freeze_20_every_20_opt_pose/test_mesh/0020_deformed.ply'), process=False)
    mesh_512 = Meshes(mesh_512.vertices, mesh_512.faces, mesh_512.vertex_normals, name='mesh_512', flat_shading=True)
    mesh_256 = Meshes(mesh_256.vertices, mesh_256.faces, mesh_256.vertex_normals, name='mesh_256', flat_shading=True)
    viewer = Viewer()
    viewer.scene.add(mesh_512)
    viewer.scene.add(mesh_256)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.run()