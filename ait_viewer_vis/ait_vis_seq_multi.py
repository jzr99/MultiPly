import os
import tqdm
import trimesh
import numpy as np

from aitviewer.viewer import Viewer
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material
import numpy as np
import cv2
if __name__ == '__main__':

    # frames_folder = "/home/chen/disk2/motion_capture/wenting/meshes"
    # DATA_DIR = "/media/ubuntu/hdd/V2A_output/data/Hi4D_pair19_piggyback19"
    # original_cameras_filename = DATA_DIR + "/cameras.npz"
    # cameras = np.load(original_cameras_filename)
    # all_files=cameras.files
    # maximal_ind=0
    # for field in all_files:
    #     maximal_ind=np.maximum(maximal_ind,int(field.split('_')[-1]))
    # num_of_cameras=maximal_ind+1
    # P0 = cameras["cam_0"]
    # out = cv2.decomposeProjectionMatrix(P0[:3, :])
    # cam_intrinsics = out[0]
    # render_R = out[1]
    # cam_center = out[2]
    # cam_center = (cam_center[:3] / cam_center[3])[:, 0]
    # render_T = -render_R @ cam_center
    # t0 = render_T
    # shift0 = render_R@(-t0)
    viewer = Viewer()
    COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0],[0.412,1.0,0.663,1.0],[0.412,0.412,0.663,1.0],[0.412,0.0,0.0,1.0],[0.0,0.0,0.663,1.0],[0.0,0.412,0.0,1.0],[1.0,0.0,0.0,1.0]]
    mean_list = []
    for p in range(5):
        vertices = []
        faces = []
        vertex_normals = []
        uvs = []
        texture_paths = []
        for idx in range(75):
            
            # mesh = trimesh.load(os.path.join(f'/media/ubuntu/hdd/V2A_output/Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose/test_mesh/{p}/{idx:04d}_deformed.ply'), process=False)
            # mesh = trimesh.load(os.path.join(f'/media/ubuntu/hdd/V2A_output/taichi01_sam_delay_depth_loop_2_MLP/test_mesh/{p}/{idx:04d}_deformed.ply'), process=False)
            # mesh = trimesh.load(os.path.join(f'/media/ubuntu/hdd/V2A_output/dance5_sam_delay_depth_loop_noshare/test_mesh/{p}/{idx:04d}_deformed.ply'), process=False)
            mesh = trimesh.load(os.path.join(f'/media/ubuntu/hdd/V2A_output/updown_sam_delay_depth_loop_noshare/test_mesh/{p}/{idx:04d}_deformed.ply'), process=False)
            if p == 0:
                mean_list.append(mesh.vertices.mean(axis=0))
            # Pi = cameras[f"cam_{idx}"]
            # out = cv2.decomposeProjectionMatrix(Pi[:3, :])
            # render_R = out[1]
            # cam_center = out[2]
            # cam_center = (cam_center[:3] / cam_center[3])[:, 0]
            # ti = -render_R @ cam_center
            # shifti = render_R@(-ti)
            # delta_shift = shifti - shift0
            # print(delta_shift)
            mesh.vertices = mesh.vertices - mean_list[idx]
            vertices.append(mesh.vertices)
            # mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
            # # vertices.append(mesh.vertices - delta_shift)
            # vertices.append(mesh.vertices - cam_center)
            faces.append(mesh.faces)
            vertex_normals.append(mesh.vertex_normals)


        m=Material(color = COLORS[p], diffuse=1.0, ambient=0.0)
        meshes = VariableTopologyMeshes(vertices,
                                        faces,
                                        vertex_normals,
                                        preload=True,
                                        color = COLORS[p],
                                        name=f"mesh_{p}",
                                        material=m,
                                        )
        # meshes.color = tuple(COLORS[p])
        # meshes.name = f"mesh_{p}"
        # meshes.material.diffuse = 1.0
        # meshes.material.ambient = 0.0
        viewer.scene.add(meshes)
    viewer.run()

    # mesh = trimesh.load(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_wo_disp_freeze_20_every_20/rendering/519.ply'), process=False)
    # ours_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='ours', flat_shading=True)

    # mesh = trimesh.load('/home/chen/SelfReconCode/data/Invisible/result/tmp.ply', process=False)
    # selfrecon_mesh = Meshes(mesh.vertices, mesh.faces, mesh.vertex_normals, name='selfrecon', flat_shading=True)

    # viewer = Viewer()
    # viewer.scene.add(ours_mesh, selfrecon_mesh)
    # viewer.run()