import os
import glob
import sys
from tqdm import tqdm
from subprocess import DEVNULL, STDOUT, check_call, run
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import multiprocessing as mp
from pathlib import Path

parser = argparse.ArgumentParser(description="This is an end to end preprocessing script for the project BrainSpy written by Anant Aggarwal")
parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Basically the parent directory of the ADNI folder")
parser.add_argument("--robex", action="store_true", help="Whether to run ROBEX Brain Extraction")
parser.add_argument("--mni_reg", action="store_true", help="Whether to run MNI Registration")
parser.add_argument("--segmentation", action="store_true", help="Whether to to segment the brain in Gray Matter, White Matter and CSF")
parser.add_argument("--fsl_install", action="store_true", help="whether to install fsl")
parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs (default: number of CPU cores)")
parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds for each processing step (default: 1800 seconds = 30 minutes)")
args = parser.parse_args()

def checkFSL():
    """Check if FSL is available and properly configured"""
    try:
        # Try multiple possible FSL paths
        fsl_paths = [
            "/root/fsl/bin/fsl",
            "/usr/local/fsl/bin/fsl", 
            "/usr/share/fsl/bin/fsl",
            "fsl"
        ]
        
        for fsl_path in fsl_paths:
            try:
                result = run([fsl_path, "version"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"FSL found at: {fsl_path}")
                    return fsl_path.replace("/bin/fsl", "")
            except:
                continue
        return None
    except Exception as e:
        print(f"Error checking FSL: {e}")
        return None

def setup_environment():
    """Setup environment for Kaggle notebooks"""
    # Set FSL environment variables
    fsl_dir = checkFSL()
    if fsl_dir:
        os.environ['FSLDIR'] = fsl_dir
        os.environ['PATH'] = f"{fsl_dir}/bin:{os.environ.get('PATH', '')}"
        print(f"FSL environment set up with FSLDIR={fsl_dir}")
    else:
        print("Warning: FSL not found. MNI registration and segmentation may fail.")
    
    # Ensure ROBEX is executable
    robex_script = Path("ROBEX/runROBEX.sh")
    if robex_script.exists():
        robex_script.chmod(0o755)
        print("ROBEX script made executable")

BASE_DIR = args.base_dir
CURRENT_DIR = os.getcwd()
commands, names = [], []

# Setup environment first
setup_environment()

if args.robex:
    def robexCommand(file, output_path):
        """ROBEX brain extraction command"""
        robex_script = os.path.join(CURRENT_DIR, "ROBEX", "runROBEX.sh")
        return [robex_script, file, output_path]
    commands.append(robexCommand)
    names.append("skull_stripped")

if args.mni_reg:
    if args.fsl_install:
        print("Installing FSL...")
        try:
            check_call(['sh', os.path.join(CURRENT_DIR, 'getfsl.sh')], stderr=STDOUT)
            print("FSL installation completed")
        except Exception as e:
            print(f"FSL installation failed: {e}")
            print("Please install FSL manually or use a Kaggle notebook with FSL pre-installed")
    
    # Get FSL directory
    fsl_dir = os.environ.get('FSLDIR', '/root/fsl')
    
    def mniCommand(file, output_path):
        """MNI152 registration command"""
        ref_template = os.path.join(fsl_dir, "data/standard/MNI152_T1_1mm_brain.nii.gz")
        # Fallback to alternative template path
        if not os.path.exists(ref_template):
            ref_template = os.path.join(fsl_dir, "data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz")
        
        return [
            os.path.join(fsl_dir, "bin/flirt"), 
            "-in", file, 
            "-ref", ref_template, 
            "-out", output_path,
            "-bins", "256", 
            "-cost", "corratio",
            "-dof", "12",
            "-omat", output_path.replace(".nii.gz", ".mat")
        ]
    commands.append(mniCommand)
    names.append("mni_registered")

if args.segmentation:
    fsl_dir = os.environ.get('FSLDIR', '/root/fsl')
    
    def segmentationCommand(file, output_path):
        """FAST segmentation command"""
        return [
            os.path.join(fsl_dir, "bin/fast"), 
            "-t", "1", 
            "-n", "3", 
            "-H", "0.1", 
            "-I", "8", 
            "-l", "20.0", 
            "-o", output_path.replace(".nii.gz", ""),
            "-B",
            "-b", file
        ]
    commands.append(segmentationCommand)
    names.append("segmented")

def process_single_file(file_info):
    """Process a single file with all specified commands"""
    file, commands, names, base_dir, current_dir, timeout = file_info
    
    try:
        # Get base output path
        base_output_path = os.path.relpath(file, base_dir)
        base_output_path = os.path.join(current_dir, base_output_path)
        
        current_file = file
        
        # Apply each command in sequence
        for i, command in enumerate(commands):
            try:
                # Create unique output path for this step
                step_name = names[i]
                output_dir = os.path.join(current_dir, step_name, os.path.dirname(os.path.relpath(file, base_dir)))
                os.makedirs(output_dir, exist_ok=True)
                
                # Create output filename with step suffix
                base_name = os.path.splitext(os.path.basename(file))[0]
                if base_name.endswith('.nii'):
                    base_name = base_name[:-4]  # Remove .nii extension
                output_filename = f"{base_name}_{step_name}.nii.gz"
                output_path = os.path.join(output_dir, output_filename)
                
                cmd = command(current_file, output_path)
                check_call(cmd, stderr=DEVNULL, stdout=DEVNULL, timeout=timeout)  # Suppress all output
                current_file = output_path
            except Exception as e:
                print(f"Error processing {file} with command {names[i]}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return False

def preprocessAndReplace(base_dir, commands, names):
    """Process all files with multiprocessing"""
    # Get all files to process
    file_pattern = os.path.join(base_dir, "ADNI/**/**/**/**/*.nii")
    files = list(glob.glob(file_pattern))
    
    if not files:
        print(f"No .nii.gz files found in {base_dir}/ADNI/**/**/**/**/")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Prepare file info for multiprocessing
    file_infos = [(file, commands, names, base_dir, CURRENT_DIR, args.timeout) for file in files]
    
    # Determine number of jobs
    if args.n_jobs:
        n_jobs = args.n_jobs
    else:
        # Use 75% of CPU cores to avoid memory/I/O bottlenecks
        n_jobs = max(1, int(mp.cpu_count() * 0.75))
        n_jobs = min(n_jobs, len(files))  # Don't exceed number of files
    
    print(f"Using {n_jobs} parallel processes (out of {mp.cpu_count()} available cores)")
    print(f"Timeout per step: {args.timeout} seconds ({args.timeout//60} minutes)")
    
    # Process files with multiprocessing
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, file_info): file_info[0] 
                         for file_info in file_infos}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"Exception occurred while processing {file}: {e}")
                    failed += 1
                pbar.update(1)
    
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    print("Running Preprocessing.....")
    preprocessAndReplace(BASE_DIR, commands, names)
    print("Preprocessing Completed.....")
