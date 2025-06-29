import os
import glob
import sys
from tqdm import tqdm

BASE_DIR = sys.argv[1]
CURRENT_DIR = os.getcwd()

ROBEX_DIR_PATH = os.path.join("BrainSpyPreprocessing", "ROBEX")
ROBEX_CALL = os.path.join(CURRENT_DIR, ROBEX_DIR_PATH, "runROBEX.sh")

def preprocessAndReplace(base_dir, command, name):
    for file in tqdm(glob.iglob(os.path.join(base_dir, "ADNI/**/**/**/**/*"))):
        output_dir = os.path.relpath(file, BASE_DIR)
        output_dir = os.path.join(name, output_dir)
        os.system(f"{command} {file} {output_dir}")

print("Running ROBEX Brain Extraction.......")
preprocessAndReplace(BASE_DIR, ROBEX_CALL, "skull_stripped")
print("Successfully ran ROBEX Brain Extraction.......")
