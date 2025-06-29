import os
import glob
import sys
from tqdm import tqdm
from subprocess import DEVNULL, STDOUT, check_call
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description="This is and end to end preprocessing script for the project BrainSpy written by Anant Aggarwal")
parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Basically the parent directory of the ADNI folder")
parser.add_argument("--robex", action="store_true", help="Whether to run ROBEX Brain Extraction")
parser.add_argument("--mni_reg", action="store_true", help="Whether to run MNI Registration")
parser.add_argument("--segmentation", action="store_true", help="Whether to to segment the brain in Gray Matter, White Matter and CSF")
parser.add_argument("--fsl_install", action="store_true", help="whether to install fsl")
args = parser.parse_args()

def checkFSL():
    try:
        check_call(["fsl", "version"], stdout=DEVNULL, stderr=STDOUT)
        return True
    except:
        return False

BASE_DIR = args.base_dir
CURRENT_DIR = os.getcwd()
commands, names = [], []

if args.robex:
    def robexCommnand(file, output_path):
        ROBEX_DIR_PATH = os.path.join("BrainSpyPreprocessing", "ROBEX")
        ROBEX_CALL = os.path.join(CURRENT_DIR, ROBEX_DIR_PATH, "runROBEX.sh")
        return [ROBEX_CALL, file, output_path]
    commands.append(robexCommnand)
    names.append("skull_stripped")

if args.mni_reg:
    if args.fsl_install:
        print("FSL is not installed")
        print("Installing FSL...")
        check_call(['sh', os.path.join(CURRENT_DIR,'BrainSpyPreprocessing', 'getfsl.sh')], stderr=STDOUT)
        check_call(['export', 'PATH=/root/fsl/bin:$PATH'], stderr=STDOUT)
        print("Successfully installed FSL.....")
    else:
        print("FSL is already installed")
    
    def mniCommand(file, output_path):
        return ["/root/fsl/bin/flirt", "-in", file, "-ref", "MNI152_T1_1mm_brain.nii.gz", "-out", output_path, 
        "-bins", "256", "-cost", "corratio","-dof", "12"]
    commands.append(mniCommand)
    names.append("mni_registered")

if args.segmentation:
    def segmentationCommand(file, output_path):
        return ["/root/fsl/bin/fast", "-t", "1", "-n", "3", "-H", "0.1", "-I", "8", "-l", "20.0", "-o", output_path,"-B","-b", file]
    commands.append(segmentationCommand)
    names.append("segmented")


def preprocessAndReplace(base_dir, commands, names):
    iters = len(glob.glob(os.path.join(base_dir, "ADNI/**/**/**/**/*")))
    for file in tqdm(glob.iglob(os.path.join(base_dir, "ADNI/**/**/**/**/*")), total=iters):
        output_path = os.path.relpath(file, BASE_DIR)
        name = '_'.join(names)
        output_path = os.path.join(CURRENT_DIR, name, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        for command in commands:
            check_call(command(file, output_path), stdout=DEVNULL, stderr=STDOUT)
            file = output_path



print("Running Preprocessing.....")
preprocessAndReplace(BASE_DIR, commands, names)
print("Preprocessing Completed.....")
