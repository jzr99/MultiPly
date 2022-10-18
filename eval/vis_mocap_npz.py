import trimesh
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pymesh

def convert_mesh_npz2ply(mesh_dir: Path, new_mesh_dir: Path):
    if not new_mesh_dir.exists():
        new_mesh_dir.mkdir(exist_ok=True)
    mesh_file_list = sorted(
        [x for x in mesh_dir.iterdir() if str(x).endswith('.npz')])
    for mesh_file in tqdm(mesh_file_list):
        mesh_data = np.load(mesh_file)
        # mesh = pymesh.meshio.form_mesh(mesh_data['vertices'],
                                    #    mesh_data['faces'])
        mesh = trimesh.Trimesh(mesh_data['vertices'], mesh_data['faces'])
        mesh.visual.uv = mesh_data['uvs']
        mesh_name = str(new_mesh_dir /
                        str(mesh_file.name).replace('.npz', '.obj'))
        # pymesh.save_mesh(mesh_name, mesh)
        _ = mesh.export(mesh_name)

if __name__ == '__main__':
    subject = '00070_Dance'
    mesh_folder = Path(f"/home/chen/disk2/RGB_PINA_MoCap/{subject}/meshes")
    output_folder = Path(f'/home/chen/disk2/RGB_PINA_MoCap/{subject}/meshes_vis')
    convert_mesh_npz2ply(mesh_folder, output_folder)