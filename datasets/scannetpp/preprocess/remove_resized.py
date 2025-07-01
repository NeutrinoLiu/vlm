from glob import glob
from tqdm import tqdm
import os
dead = glob("/mnt/bn/nlhei-nas/liubangya/proj/vlm/datasets/scannetpp/scannetpp-data/data/*/dslr/resized_*")
dead.sort()
for d in tqdm(dead):
    os.system(f"rm -rf {d}")