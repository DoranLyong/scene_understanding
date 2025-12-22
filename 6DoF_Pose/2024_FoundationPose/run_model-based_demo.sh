export CUDA_VISIBLE_DEVICES="0"

#python run_demo.py


# >> Run on public datasets 
#LINEMOD_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/linemod
#python run_linemod.py --linemod_dir $LINEMOD_DIR --use_reconstructed_mesh 0


YCBV_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/ycbv
python run_ycb_video.py --ycbv_dir $YCBV_DIR --use_reconstructed_mesh 0
