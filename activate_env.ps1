# Local Python environment for this checkout.
#
# The repository dependencies are unpacked into .pylibs/ (see .wheelhouse/) because
# this machine's global site-packages is not writable and the sandbox blocks pip's
# temp-file install path. Dot-source this file, then run the README commands as-is:
#
#   . .\activate_env.ps1
#   python models\train.py --data datasets\RML2016.10a_dict.dat `
#       --split datasets\split_indices.npz --epochs 100 `
#       --output models\checkpoints\dscamcnet_best.pt

$env:PYTHONPATH = (Join-Path $PSScriptRoot ".pylibs")
$env:MPLCONFIGDIR = (Join-Path $PSScriptRoot ".mplconfig")

if (-not (Test-Path $env:MPLCONFIGDIR)) { New-Item -ItemType Directory -Path $env:MPLCONFIGDIR | Out-Null }

python -c "import torch, onnx, onnxruntime, scipy, sklearn; print('env ready: torch', torch.__version__, '| onnx', onnx.__version__, '| onnxruntime', onnxruntime.__version__)"
