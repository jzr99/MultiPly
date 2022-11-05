import numpy as np
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

smpl_layer = SMPLLayer(model_type='smpl',gender='male')

poses = np.load('/home/chen/RGB-PINA/data/Invisible/poses.npy')
betas = np.load('/home/chen/RGB-PINA/data/Invisible/mean_shape.npy')
smpl = SMPLSequence(poses_body=poses[:1, 3:],
                    smpl_layer=smpl_layer,
                    poses_root=poses[:1, :3],
                    betas=betas,
                    color=(0.0, 106 / 255, 139 / 255, 1.0),
                    name='Refined Estimate')

viewer = Viewer()
viewer.scene.add(smpl)
viewer.scene.floor.enabled = False
viewer.shadows_enabled = False
viewer.scene.origin.enabled = False

viewer.run()