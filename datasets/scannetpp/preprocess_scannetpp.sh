cd /mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp/scannetpp-devkit
python -m dslr.undistort /mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp/preprocess/undistort_330.yml

# Configuration file located at vlm_proj/scannetpp/semantic/configs/rasterize.yaml
# 
# To adjust image resolution, modify `image_downsample_factor`.
# To change batch size during rasterization, update `batch_size`.
# Note: `subsample_factor` controls frame interval sampling (not resolution); it is typically best to leave it unchanged.
python -m semantic.prep.rasterize --config-path=/mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp/preprocess/ --config-name=rasterize

python -m semantic.prep.rasterize_mp --config-path=/mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp/preprocess/ --config-name=rasterize_4GPU


cd /mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp
python structure-data-scannetpp.py 

# conda activate qwen
# python caption_crop.py
# python templating-naive.py