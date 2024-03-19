# pre-define data
# source="custom"
# seq="dance4"
# gender="FEMALE"
# number=4

# source="iphone"
# source="custom"
# seq="sydney"
# gender="FEMALE"
# number=5

# source="custom"
# seq="taichi_single3"
# gender="FEMALE"
# number=2

folder_path="/media/ubuntu/hdd/RGB-PINA/preprocessing" # absolute path of preprocessing folder
source="custom"
seq="piggy19_cam4"
seq_path="/home/ubuntu/Downloads/$seq\.mp4"
gender="FEMALE"
number=2
time_start="00:00:00"
time_duration="00:00:15"

cd /media/ubuntu/hdd/RGB-PINA/preprocessing/

# run ROMP to get initial SMPL parameters
echo "Running Trace"
conda activate smoothnet-env
mkdir ./raw_data/$seq
mkdir ./raw_data/$seq/$seq
# ffmpeg -i /home/ubuntu/Downloads/$seq\.mp4 -vsync 0 /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq/%04d.png
ffmpeg -i $seq_path -ss $time_start -t $time_duration -vsync 0 ./raw_data/$seq/$seq/%04d.png

# # cut off depth image
# conda activate v2a_global
# cd /media/ubuntu/hdd/RGB-PINA/preprocessing/
# python cut_half.py --seq $seq 
# conda activate smoothnet-env

# trace2 -i '/media/ubuntu/hdd/ROMP/trace_video/dance01/triple02_dance02_88_clip'  --subject_num=3 --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40
# trace2 -i '/media/ubuntu/hdd/ROMP/trace_video/$seq'  --subject_num=$number --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40
# trace2 -i /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq  --subject_num=$number --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40
trace2 -i $folder_path/raw_data/$seq/$seq  --subject_num=$number --results_save_dir=./trace_results/ --save_video --show_tracking --time2forget=40
# copy img to frames
# mkdir /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq
# cp -r /home/ubuntu/TRACE_results/$seq\_frames /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/frames
mv ./raw_data/$seq/$seq ./raw_data/$seq/frames

echo "reformate the data"
conda activate aitviewer
# cd /media/ubuntu/hdd/ROMP
# python /media/ubuntu/hdd/ROMP/aitcamera.py --seq $seq
python ../ait_viewer_vis/aitcamera.py --seq $seq --headless

# obtain the projected masks through estimated perspective camera (so that OpenPose detection)
echo "Getting projected SMPL masks"
conda activate v2a_global
# python preprocessing_multiple_trace.py --source $source --seq $seq --gender $gender --mode mask
# python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode mask
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
# python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose --openpose
python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose

# scale images and center the human in 3D space
echo "Scaling images and centering human in 3D space"
# python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode final
python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 2
# python preprocessing.py --source $source --seq $seq --gender $gender --mode final --scale_factor 1

# normalize cameras such that all cameras are within the sphere of radius 3
echo "Normalizing cameras"
# python normalize_cameras.py --input_cameras_file ../data/$seq/cameras.npz \
#                             --output_cameras_file ../data/$seq/cameras_normalize.npz
python normalize_cameras_trace.py --input_cameras_file ../data/$seq/cameras.npz \
                            --output_cameras_file ../data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file ../data/$seq/max_human_sphere.npy

# scp -r /media/ubuntu/hdd/RGB-PINA/data/$seq jiangze@euler.ethz.ch:/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/
