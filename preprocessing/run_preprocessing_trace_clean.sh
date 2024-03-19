folder_path="/media/ubuntu/hdd/RGB-PINA/preprocessing" # absolute path of preprocessing folder
source="custom"
seq="piggy19_cam4" # name of the sequence
seq_path="/home/ubuntu/Downloads/$seq\.mp4" # path to the seq
number=2
time_start="00:00:00"
time_duration="00:00:15"

cd /media/ubuntu/hdd/RGB-PINA/preprocessing/

# run ROMP to get initial SMPL parameters
echo "Running Trace"
conda activate smoothnet-env
mkdir ./raw_data/$seq
mkdir ./raw_data/$seq/$seq
ffmpeg -i $seq_path -ss $time_start -t $time_duration -vsync 0 ./raw_data/$seq/$seq/%04d.png

# estimate the SMPL parameter and tracking with trace
trace2 -i $folder_path/raw_data/$seq/$seq  --subject_num=$number --results_save_dir=./trace_results/ --save_video --show_tracking --time2forget=40
mv ./raw_data/$seq/$seq ./raw_data/$seq/frames

echo "reformate the data"
conda activate aitviewer
python ../ait_viewer_vis/aitcamera.py --seq $seq --headless

# obtain the projected masks through estimated perspective camera (so that OpenPose detection)
echo "Getting projected SMPL masks"
conda activate v2a_global
cd /media/ubuntu/hdd/RGB-PINA/preprocessing/
python preprocessing_multiple_trace.py --source custom --seq $seq --mode mask

# run OpenPose to get 2D keypoints
# echo "Running OpenPose"
# python run_openpose_multiple_trace.py --openpose_dir /media/ubuntu/hdd/openpose --seq $seq

# run vitpose to get 2D keypoints
conda activate vitpose
python ./vitpose_trace.py --img-root $folder_path/raw_data/$seq/frames --kpt-thr 0.3
conda activate v2a_global

# offline refine poses
echo "Refining poses offline"
python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose
# optional openpose refine
# python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose --openpose

# scale images and center the human in 3D space
echo "Scaling images and centering human in 3D space"
python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 2

# normalize cameras such that all cameras are within the sphere of radius 3
echo "Normalizing cameras"
python normalize_cameras_trace.py --input_cameras_file ../data/$seq/cameras.npz \
                            --output_cameras_file ../data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file ../data/$seq/max_human_sphere.npy

# scp -r /media/ubuntu/hdd/RGB-PINA/data/$seq jiangze@euler.ethz.ch:/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/
