# train.ps1
# Runs VisionForge training using the venv Python directly.
# This bypasses `uv run` which would revert torch to the CPU version from uv.lock.
# Usage:  .\train.ps1 configs\synthetic_test.yaml
#         .\train.ps1 configs\baseline.yaml

param(
    [Parameter(Mandatory=$true)]
    [string]$Config
)

$python = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python — run 'uv sync' first."
    exit 1
}

& $python -m visionforge $Config
