



conda activate nusc
cd scannetpp
python -m dslr.undistort dslr/configs/undistort.yml

# Configuration file located at vlm_proj/scannetpp/semantic/configs/rasterize.yaml
# 
# To adjust image resolution, modify `image_downsample_factor`.
# To change batch size during rasterization, update `batch_size`.
# Note: `subsample_factor` controls frame interval sampling (not resolution); it is typically best to leave it unchanged.
python -m semantic.prep.rasterize
cd ..


cd QA_scannetpp/
python structure-data-scannetpp.py 

conda activate qwen
python caption_crop.py
python templating-naive.py