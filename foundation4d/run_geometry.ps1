param([int]$Epochs=8,[int]$Workers=2)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    & "$root/.venv/Scripts/python.exe" -u -m foundation4d.train --project-root $root --epochs $Epochs --workers $Workers --image-size 480 864 --query-chunk 1024 --geometry-aware --depth-weight 1 --boundary-weight 0.1 --ray-weight 0.05 --height-balance-cap 1.5 --output-dir foundation4d/outputs/lite_480_geometry_seed0
    if($LASTEXITCODE -ne 0){throw "Training exited with code $LASTEXITCODE"}
} finally {Pop-Location}
