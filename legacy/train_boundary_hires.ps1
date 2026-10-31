$ErrorActionPreference = 'Stop'
$legacyProjectRoot = Split-Path $PSScriptRoot -Parent
Push-Location $legacyProjectRoot
try {
    $experimentOutput = Join-Path $legacyProjectRoot 'outputs\train_bevdepth_boundary_hires_train100'
    if (Test-Path -LiteralPath $experimentOutput) {
        throw "Output already exists. Resume its latest.pt instead of overwriting: $experimentOutput"
    }
    $trainingArgs = @(
        '-u', 'train.py', '--architecture', 'bevdepth', '--depth', '--xyz',
        '--image-size', '480', '864', '--geometry-loss-weight', '0.2',
        '--boundary-loss-weight', '0.1', '--depth-target-mode', 'projected',
        '--depth-loss-mode', 'bce', '--depth-loss-weight', '3',
        '--dataset-version', 'v1.0-trainval', '--dataroot', 'data/occ3d_trainval',
        '--annotations-json', 'data/occ3d_trainval/downloads/annotations.json',
        '--occupancy-root', 'data/occ3d_trainval/gts',
        '--train-scenes', 'outputs/splits/train100_val20/train_scenes.json',
        '--val-scenes', 'outputs/splits/train100_val20/val_scenes.json',
        '--epochs', '5', '--overfit-samples', '0', '--class-balance', 'sqrt-inverse',
        '--batch-size', '1', '--workers', '2', '--seed', '0',
        '--output-dir', $experimentOutput
    )
    & '.\.venv\Scripts\python.exe' @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
