# BrainSpy Preprocessing for Kaggle Notebooks

This repository contains preprocessing tools for brain MRI data, optimized for running on Kaggle notebooks with multiprocessing support.

## Features

- **ROBEX Brain Extraction**: Automatic skull stripping
- **MNI152 Registration**: Standard space registration using FSL FLIRT
- **FAST Segmentation**: Tissue segmentation (GM, WM, CSF)
- **Multiprocessing**: Parallel processing for faster execution
- **Kaggle Optimized**: Works seamlessly in Kaggle notebook environment

## Quick Start

### 1. Clone the Repository

```bash
!git clone https://github.com/your-username/BrainSpyPreprocessing.git
%cd BrainSpyPreprocessing
```

### 2. Run Setup

```bash
!python setup_kaggle.py
```

### 3. Run Preprocessing

```bash
!python preprocess.py \
    --base_dir /kaggle/input/your-dataset \
    --robex \
    --mni_reg \
    --segmentation \
    --n_jobs 4
```

## Dataset Structure

Your dataset should follow this structure:
```
your-dataset/
└── ADNI/
    └── [subject_folders]/
        └── [session_folders]/
            └── [modality_folders]/
                └── *.nii.gz
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--base_dir` | Base directory containing ADNI folder | Required |
| `--robex` | Run ROBEX brain extraction | False |
| `--mni_reg` | Run MNI152 registration | False |
| `--segmentation` | Run FAST segmentation | False |
| `--fsl_install` | Install FSL (if not present) | False |
| `--n_jobs` | Number of parallel processes | CPU count |

## Examples

### ROBEX Only (Brain Extraction)
```bash
!python preprocess.py --base_dir /kaggle/input/your-dataset --robex
```

### MNI Registration Only
```bash
!python preprocess.py --base_dir /kaggle/input/your-dataset --mni_reg
```

### Segmentation Only
```bash
!python preprocess.py --base_dir /kaggle/input/your-dataset --segmentation
```

### All Steps with Custom Parallel Jobs
```bash
!python preprocess.py \
    --base_dir /kaggle/input/your-dataset \
    --robex \
    --mni_reg \
    --segmentation \
    --n_jobs 8
```

## Output Structure

The processed files will be saved in folders named after the processing steps:

```
current_directory/
├── skull_stripped/          # ROBEX output
├── mni_registered/          # MNI registration output
├── segmented/               # FAST segmentation output
└── skull_stripped_mni_registered_segmented/  # Combined processing
```

## Requirements

### Python Packages
- `tqdm` - Progress bars
- `nibabel` - Neuroimaging file handling
- `numpy` - Numerical computations

### System Dependencies
- **FSL** - For MNI registration and segmentation
- **ROBEX** - For brain extraction (included in repo)

## Troubleshooting

### FSL Not Found
If FSL is not available, you can install it:
```bash
!apt-get update && apt-get install -y fsl
```

### Memory Issues
Reduce the number of parallel jobs:
```bash
!python preprocess.py --base_dir /kaggle/input/your-dataset --robex --n_jobs 2
```

### Timeout Issues
The script includes a 5-minute timeout per file. For very large files, you may need to increase this in the code.

## Performance Tips

1. **Use appropriate number of jobs**: Start with `--n_jobs 4` and adjust based on your Kaggle instance
2. **Process in batches**: For large datasets, consider processing subsets
3. **Monitor memory usage**: Use Kaggle's resource monitor to avoid OOM errors

## File Formats

- **Input**: NIfTI (.nii.gz) files
- **Output**: NIfTI (.nii.gz) files
- **Transformation matrices**: .mat files (for MNI registration)

## License

[Add your license information here]

## Citation

If you use this preprocessing pipeline, please cite:
[Add citation information here] 