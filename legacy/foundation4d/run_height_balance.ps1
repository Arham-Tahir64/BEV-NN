param([int]$Epochs=8,[int]$Workers=2)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Push-Location $root
try {
    & "$root/.venv/Scripts/python.exe" -u -m foundation4d.train --project-root $root --epochs $Epochs --workers $Workers --image-size 480 864 --query-chunk 4096 --height-balance-cap 1.5 --output-dir foundation4d/outputs/lite_480_heightcap1p5_seed0
    if($LASTEXITCODE -ne 0){throw "Training exited with code $LASTEXITCODE"}
} finally {Pop-Location}
