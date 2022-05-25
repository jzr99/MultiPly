# produce visible meshes
echo "produce visible meshes"
python produce_vis_mesh.py --capture_root /home/chen/disk2/motion_capture/otmar_waving

# undistort images
echo "undistort images"
python undistort_images.py --capture_root /home/chen/disk2/motion_capture/otmar_waving

# run openpose
echo "run openpose"
python pose2d_single.py --capture_root /home/chen/disk2/motion_capture/otmar_waving --openposefolder /home/chen/openpose

# triangulate
echo "triangulate"
python pose3d_single.py --capture_root /home/chen/disk2/motion_capture/otmar_waving

