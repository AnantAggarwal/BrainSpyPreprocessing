#!/usr/bin/env sh
#
# getfsl.sh script for installing FSL in one step. This script
# is only intended for typical single-user FSL installations
# into (e.g.) ~/fsl/. For more complicated installations (e.g.
# on mullti-user systems), please download and use the
# fslinstaller.py script directly - you can downlaod the
# fslinstaller.py script from:
#
#   https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
#
# Usage: getfsl.sh [fsldir [fslinstaller.py arguments]]
#

set -e


##############################################
# Variables set when this script is generated.
##############################################


# Micromamba URLs
MM_LINUX_64_URL="https://anaconda.org/conda-forge/micromamba/2.0.5/download/linux-64/micromamba-2.0.5-0.tar.bz2"
MM_MACOS_64_URL="https://anaconda.org/conda-forge/micromamba/2.0.5/download/osx-64/micromamba-2.0.5-0.tar.bz2"
MM_MACOS_M1_URL="https://anaconda.org/conda-forge/micromamba/2.0.5/download/osx-arm64/micromamba-2.0.5-0.tar.bz2"

# Python verson to install
PYTHON_VERSION="3.12.*"

# fslinstaller.py URL
INSTALLER_URL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py"

###################################################
# Check that necessary UNIX commands are available.
# We need curl or wget, tar, and bzip2.
###################################################

# Pre-canned command to download
# a file to standard output with
# either wget or curl
if command -v curl > /dev/null; then
  DOWNLOAD_COMMAND="curl -L -s -S -f"
elif command -v wget > /dev/null; then
  DOWNLOAD_COMMAND="wget -L -q -O -"
else
  echo "The getfsl.sh script requires the curl or wget command!"
  exit 1
fi

if ! command -v tar > /dev/null; then
  echo "The getfsl.sh script requires the tar command!"
  exit 1
fi

if ! command -v bzip2 > /dev/null; then
  echo "The getfsl.sh script requires the bzip2 command!"
  exit 1
fi


##########################################
# Determine the FSL installation location.
##########################################


# User specified fsldir as argument?
if [ "$#" -gt "0" ]; then
  FSLDIR="${1}"
  shift

# $FSLDIR variable already set?
elif [ "${FSLDIR}" != "" ]; then
  FSLDIR=${FSLDIR}

# Otherwise default to ~/fsl/
else
  FSLDIR="${HOME}/fsl"
fi

# Make sure $FSLDIR is an absolute path
case "${FSLDIR}" in
  /*) FSLDIR="${FSLDIR}" ;;
  *)  FSLDIR="${PWD}/${FSLDIR}" ;;
esac


##########################################
# Check to see if $FSLDIR already exists.
# The user must remove an existing FSL
# installation before running this script.
##########################################


if [ -e ${FSLDIR} ]; then
  echo "Installation directory ${FSLDIR} already exists!" \
       "Remove ${FSLDIR}, or choose a different location," \
       "and then re-run this script."
  exit 1
fi


##########################################
# Check to see if the user has permission
# to create $FSLDIR
##########################################


can_create_dir() {
  target="${1}"

  while [ ! -d "${target}" ]; do
    target=$(dirname "${target}")
  done

  [ -w "${target}" ] && [ -x "${target}" ]
}


if ! can_create_dir "${FSLDIR}"; then
  echo "You do not have permission to create ${FSLDIR}!" \
       "The recommended approach is to install FSL into a" \
       "location that does not require root privileges," \
       "such as ${HOME}/fsl/."
  echo ""
  echo "If you do need to install FSL into a location that" \
       "requires root privileges, please download and run the" \
       "fslinstaller.py script directly. You can download the" \
       "fslinstaller.py script from ${INSTALLER_URL}."
  exit 1
fi


# Identify the platform and choose a
# suitable micromamba installer
UNAME="$(uname -a)"
case ${UNAME} in
  *Linux*x86_64*)
    MM_URL=${MM_LINUX_64_URL}
    ;;
  *Darwin*x86_64)
    MM_URL=${MM_MACOS_64_URL}
    ;;
  *Darwin*arm64*)
    MM_URL=${MM_MACOS_M1_URL}
    ;;
  *)
    echo "Cannot identify platform!"
    echo ""
    echo "Re-run this script and specify your platform via "
    echo "an environment variable called FSLPLATFORM, e.g.:"
    echo ""
    echo "For Linux (Intel):"
    echo "  FSLPLATFORM=linux-64 sh getfsl.sh [fsldir]"
    echo ""
    echo "For macOS (Intel):"
    echo "  FSLPLATFORM=macos-64 sh getfsl.sh [fsldir]"
    echo ""
    echo "For macOS (Apple Silicon):"
    echo "  FSLPLATFORM=macos-M1 sh getfsl.sh [fsldir]"
    exit 1
esac


echo "Creating FSLDIR at ${FSLDIR} ..."

mkdir -p ${FSLDIR}


echo "Downloading Micromamba ..."

${DOWNLOAD_COMMAND} ${MM_URL} | \
  tar -C ${FSLDIR} -xj bin/micromamba


echo "Installing Python ..."

export MAMBA_ROOT_PREFIX=${FSLDIR}
${FSLDIR}/bin/micromamba install -y -q \
  -c conda-forge                         \
  -p ${FSLDIR}                         \
  python="${PYTHON_VERSION}"


echo "Installing FSL ..."

${DOWNLOAD_COMMAND} ${INSTALLER_URL} | \
  ${FSLDIR}/bin/python -                 \
  --miniconda ${FSLDIR}                  \
  --dest ${FSLDIR}                       \
  "$@"

echo ""
echo "FSL installation completed successfully!"
echo ""

# Create FSL environment setup script
FSL_SETUP_SCRIPT="${FSLDIR}/etc/fslconf/fsl.sh"

# Ensure the directory exists
mkdir -p "$(dirname "${FSL_SETUP_SCRIPT}")"

# Create the FSL setup script
cat > "${FSL_SETUP_SCRIPT}" << 'EOF'
#!/bin/bash
# FSL Environment Setup Script
# This script sets up the environment for FSL commands

# Set FSLDIR if not already set
if [ -z "${FSLDIR}" ]; then
    export FSLDIR="$(dirname "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")")"
fi

# Add FSL bin directory to PATH
if [[ ":$PATH:" != *":${FSLDIR}/bin:"* ]]; then
    export PATH="${FSLDIR}/bin:$PATH"
fi

# Add FSL lib directory to LD_LIBRARY_PATH (Linux) or DYLD_LIBRARY_PATH (macOS)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if [[ ":$LD_LIBRARY_PATH:" != *":${FSLDIR}/lib:"* ]]; then
        export LD_LIBRARY_PATH="${FSLDIR}/lib:$LD_LIBRARY_PATH"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ ":$DYLD_LIBRARY_PATH:" != *":${FSLDIR}/lib:"* ]]; then
        export DYLD_LIBRARY_PATH="${FSLDIR}/lib:$DYLD_LIBRARY_PATH"
    fi
fi

# Set FSL output type
export FSLOUTPUTTYPE=NIFTI_GZ

# Set FSL configuration directory
export FSLMULTIFILEQUIT=TRUE
export FSLCONFDIR="${FSLDIR}/etc/fslconf"

# Set FSL data directory
export FSLOUTPUTTYPE=NIFTI_GZ

# Set FSL environment variables for better performance
export FSLTCLSH="${FSLDIR}/bin/fsltclsh"
export FSLWISH="${FSLDIR}/bin/fslwish"

# Set FSL environment for conda installation
export CONDA_PREFIX="${FSLDIR}"
export MAMBA_ROOT_PREFIX="${FSLDIR}"

# Initialize conda environment
if [ -f "${FSLDIR}/etc/profile.d/conda.sh" ]; then
    . "${FSLDIR}/etc/profile.d/conda.sh"
    conda activate "${FSLDIR}"
fi

echo "FSL environment configured. FSLDIR=${FSLDIR}"
echo "You can now use FSL commands like flirt, fast, bet, etc."
EOF

# Make the setup script executable
chmod +x "${FSL_SETUP_SCRIPT}"

# Create a user-friendly setup script in the FSL root directory
USER_SETUP_SCRIPT="${FSLDIR}/setup_fsl.sh"
cat > "${USER_SETUP_SCRIPT}" << EOF
#!/bin/bash
# Quick FSL Environment Setup
# Source this script to set up FSL environment in your current shell

source "${FSL_SETUP_SCRIPT}"

echo ""
echo "FSL environment is now active!"
echo "Try running: flirt -help"
echo "Or: fast -help"
echo ""
EOF

chmod +x "${USER_SETUP_SCRIPT}"

echo "=========================================="
echo "FSL Installation Complete!"
echo "=========================================="
echo ""
echo "To use FSL commands (flirt, fast, bet, etc.), you need to set up the environment:"
echo ""
echo "Option 1 - For current shell session:"
echo "  source ${USER_SETUP_SCRIPT}"
echo ""
echo "Option 2 - For permanent setup, add to your ~/.bashrc or ~/.zshrc:"
echo "  echo 'source ${USER_SETUP_SCRIPT}' >> ~/.bashrc"
echo ""
echo "Option 3 - Run FSL commands directly with full path:"
echo "  ${FSLDIR}/bin/flirt -help"
echo "  ${FSLDIR}/bin/fast -help"
echo ""
echo "After setting up the environment, you can test with:"
echo "  flirt -help"
echo "  fast -help"
echo ""