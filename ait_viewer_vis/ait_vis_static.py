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


    mesh_512 = trimesh.load(os.path.join('/home/chen/V2A_release/RGB-PINA/code/outputs/ThreeDPW/parkinglot_release_new_loss_bg_frame/test_mesh/0033_canonical.ply'), process=False)
    mesh_512 = Meshes(mesh_512.vertices, mesh_512.faces, mesh_512.vertex_normals, name='mesh_512', flat_shading=True)
    viewer = Viewer()
    viewer.scene.add(mesh_512)
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = True
    viewer.run()