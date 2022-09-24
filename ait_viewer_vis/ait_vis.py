import os
import tqdm
import trimesh
import numpy as np

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

    for idx in range(100):
        mesh = trimesh.load(os.path.join(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/outdoors_fencing_01_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled/{idx:04d}_deformed.ply'), process=False)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces)
        vertex_normals.append(mesh.vertex_normals)



    meshes = VariableTopologyMeshes(vertices,
                                    faces,
                                    vertex_normals,
                                    preload=True
                                    )

    viewer = Viewer()
    viewer.scene.add(meshes)
    viewer.run()

    # mesh = trimesh.load(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_wo_disp_freeze_20_every_20/rendering/519.ply'), process=False)
    # ours_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='ours', flat_shading=True)

    # mesh = trimesh.load('/home/chen/SelfReconCode/data/Invisible/result/tmp.ply', process=False)
    # selfrecon_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='selfrecon', flat_shading=True)

    # viewer = Viewer()
    # viewer.scene.add(ours_mesh, selfrecon_mesh)
    # viewer.run()