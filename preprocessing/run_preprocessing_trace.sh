# pre-define data
source="custom"
seq=""
gender="FEMALE"

# run ROMP to get initial SMPL parameters
echo "Running Trace"
trace2 -i '/media/ubuntu/hdd/ROMP/trace_video/dance01/triple02_dance02_88_clip'  --subject_num=3 --results_save_dir=./trace_results --save_video --show_tracking --time2forget=40

echo "reformate the data"
cd /media/ubuntu/hdd/ROMP
python aitcamera.py

# obtain the projected masks through estimated perspective camera (so that OpenPose detection)
echo "Getting projected SMPL masks"
# python preprocessing_multiple_trace.py --source $source --seq $seq --gender $gender --mode mask
python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode mask

# run OpenPose to get 2D keypoints
echo "Running OpenPose"
# python run_openpose.py --openpose_dir {PATH_TO_OPENPOSE} --seq $seq
python run_openpose_multiple_trace.py --openpose_dir /media/ubuntu/hdd/openpose --seq triple02_dance02_88

# offline refine poses
echo "Refining poses offline"
# python preprocessing.py --source $source --seq $seq --gender $gender --mode refine
python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode refine

# scale images and center the human in 3D space
echo "Scaling images and centering human in 3D space"
python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode final
# python preprocessing.py --source $source --seq $seq --gender $gender --mode final --scale_factor 1

# normalize cameras such that all cameras are within the sphere of radius 3
echo "Normalizing cameras"
python normalize_cameras.py --input_cameras_file ../data/$seq/cameras.npz \
                            --output_cameras_file ../data/$seq/cameras_normalize.npz