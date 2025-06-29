import os
import glob
import sys
from tqdm import tqdm

BASE_DIR = sys.argv[1]

ROBEX_DIR_PATH = ""
ROBEX_CALL = os.join(ROBEX_DIR_PATH, "runROBEX.sh")

def preprocessAndReplace(base_dir, command):
    for file in tqdm(glob.iglob(os.join(base_dir, "ADNI/**/**/**/**/*"))):
        os.system(f"{command} {file} {file}")

print("Running ROBEX Brain Extraction.......")
preprocessAndReplace(BASE_DIR, ROBEX_CALL)