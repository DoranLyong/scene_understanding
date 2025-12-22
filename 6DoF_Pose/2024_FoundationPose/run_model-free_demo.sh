export CUDA_VISIBLE_DEVICES="0"



# >> Data prepare  for ref_view_dir 
YCBV_REF_DIR=/home/cvipl-ubuntu/Workspace/datasets/6DoF/ycbv/ref_views_16


python bundlesdf/run_nerf.py --ref_view_dir $YCBV_REF_DIR \
                            --dataset ycbv
