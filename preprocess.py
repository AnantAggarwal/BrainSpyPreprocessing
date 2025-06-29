import os
import glob
import sys
from tqdm import tqdm
from subprocess import DEVNULL, STDOUT, check_call

BASE_DIR = sys.argv[1]
CURRENT_DIR = os.getcwd()

ROBEX_DIR_PATH = os.path.join("BrainSpyPreprocessing", "ROBEX")
ROBEX_CALL = os.path.join(CURRENT_DIR, ROBEX_DIR_PATH, "runROBEX.sh")

def preprocessAndReplace(base_dir, command, name):
    for file in tqdm(glob.iglob(os.path.join(base_dir, "ADNI/**/**/**/**/*"))):
        output_path = os.path.relpath(file, BASE_DIR)
        output_path = os.path.join(name, output_path)
        check_call([command, file, output_path, stdout=DEVNULL, stderr=STDOUT)
        print(f"Created {output_path} from {file}")

print("Running ROBEX Brain Extraction.......")
preprocessAndReplace(BASE_DIR, ROBEX_CALL, "skull_stripped")
print("Successfully ran ROBEX Brain Extraction.......")
