# pre-define data
# pair="pair17"
# action="piggyback17"
# camera="40"
# source="hi4d"
# seq="pair17_piggyback17_40"
# gender="FEMALE"
# number=2

# pair="pair15"
# action="fight15"
# camera="44"
# source="hi4d"
# seq="pair15_fight15_test_4"
# gender="FEMALE"
# number=2

# pair="pair16"
# action="jump16"
# camera="4"
# source="hi4d"
# seq="pair16_jump16_vitpose_4"
# gender="FEMALE"
# number=2

# pair="pair19"
# action="piggyback19"
# camera="4"
# source="hi4d"
# seq="pair19_piggyback19_vitpose_4"
# gender="FEMALE"
# number=2

# pair="pair15"
# action="fight15"
# camera="4"
# source="hi4d"
# seq="pair15_fight15_vitpose_4"
# gender="FEMALE"
# number=2

# pair="pair17"
# action="dance17"
# camera="28"
# source="hi4d"
# seq="pair17_dance17_vitpose_28"
# gender="FEMALE"
# number=2

pair="pair18"
action="basketball18"
camera="4"
source="hi4d"
seq="pair18_basketball18_vitpose_4"
gender="FEMALE"
number=2

# run ROMP to get initial SMPL parameters
echo "Running Trace"
conda activate smoothnet-env
mkdir /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq
mkdir /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq
cp -r /media/ubuntu/hdd/Hi4D/$pair/$action/images/$camera/* /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq
cp -r /media/ubuntu/hdd/Hi4D/$pair/$action/cameras /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq
# ffmpeg -i /home/ubuntu/Downloads/$seq\.mp4 -ss 00:00:12 -t 00:00:14 -vsync 0 /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq/%04d.png
# trace2 -i '/media/ubuntu/hdd/ROMP/trace_video/dance01/triple02_dance02_88_clip'  --subject_num=3 --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40
# trace2 -i '/media/ubuntu/hdd/ROMP/trace_video/$seq'  --subject_num=$number --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40
trace2 -i /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq  --subject_num=$number --results_save_dir=/media/ubuntu/hdd/ROMP/simple_romp/trace_results/ --save_video --show_tracking --time2forget=40

# copy img to frames
# mkdir /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq
# cp -r /home/ubuntu/TRACE_results/$seq\_frames /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/frames
mv /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/$seq /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/frames

echo "reformate the data"
conda activate aitviewer
# cd /media/ubuntu/hdd/ROMP
python /media/ubuntu/hdd/ROMP/aitcamera.py --seq $seq

# obtain the projected masks through estimated perspective camera (so that OpenPose detection)
echo "Getting projected SMPL masks"
conda activate v2a_global
# python preprocessing_multiple_trace.py --source $source --seq $seq --gender $gender --mode mask
# python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode mask
cd /media/ubuntu/hdd/RGB-PINA/preprocessing/
python preprocessing_multiple_trace.py --source $source --seq $seq --mode mask

# run OpenPose to get 2D keypoints
echo "Running OpenPose"
# python run_openpose.py --openpose_dir {PATH_TO_OPENPOSE} --seq $seq
# python run_openpose_multiple_trace.py --openpose_dir /media/ubuntu/hdd/openpose --seq triple02_dance02_88
# python run_openpose_multiple_trace.py --openpose_dir /media/ubuntu/hdd/openpose --seq $seq
conda activate vitpose
cd /media/ubuntu/hdd/ViTPose/ViTPose
python demo/top_down_img_demo_with_mmdet_trace.py --img-root /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq/frames
cd /media/ubuntu/hdd/RGB-PINA/preprocessing/
conda activate v2a_global


# offline refine poses
echo "Refining poses offline"
# python preprocessing.py --source $source --seq $seq --gender $gender --mode refine
# python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode refine
# python preprocessing_multiple_trace.py --source $source --seq $seq --mode refine
python preprocessing_multiple_trace.py --source $source --seq $seq --mode refine --vitpose


# scale images and center the human in 3D space
echo "Scaling images and centering human in 3D space"
# python preprocessing_multiple_trace.py --source custom --seq triple02_dance02_88 --mode final
python preprocessing_multiple_trace.py --source $source --seq $seq --mode final --scale_factor 2
# python preprocessing.py --source $source --seq $seq --gender $gender --mode final --scale_factor 1

# normalize cameras such that all cameras are within the sphere of radius 3
echo "Normalizing cameras"
# python normalize_cameras.py --input_cameras_file ../data/$seq/cameras.npz \
#                             --output_cameras_file ../data/$seq/cameras_normalize.npz
python normalize_cameras_trace.py --input_cameras_file ../data/$seq/cameras.npz \
                            --output_cameras_file ../data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file ../data/$seq/max_human_sphere.npy

sshpass -p Cody01066324763! scp -r /media/ubuntu/hdd/RGB-PINA/data/$seq jiangze@euler.ethz.ch:/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/

sshpass -p Cody01066324763! scp -r /media/ubuntu/hdd/RGB-PINA/preprocessing/raw_data/$seq jiangze@euler.ethz.ch:/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/preprocessing/raw_data/
