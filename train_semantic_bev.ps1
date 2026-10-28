param(
    [ValidateSet('baseline', 'dice', 'bev', 'both')][string]$Variant = 'dice',
    [ValidateRange(1, 10000)][int]$Epochs = 5,
    [ValidateRange(0, 64)][int]$Workers = 2,
    [ValidateRange(0, 2147483647)][int]$Seed = 0
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $experimentOutput = Join-Path $PSScriptRoot "outputs\semantic_bev_${Variant}_seed${Seed}"
    if (Test-Path -LiteralPath $experimentOutput) {
        throw "Output already exists. Resume its latest.pt instead of overwriting: $experimentOutput"
    }
    $diceWeight = if ($Variant -in @('dice', 'both')) { '0.2' } else { '0' }
    $bevWeight = if ($Variant -in @('bev', 'both')) { '0.1' } else { '0' }
    $trainingArgs = @(
        '-u', 'train.py', '--architecture', 'bevdepth', '--depth', '--xyz',
        '--image-size', '480', '864', '--geometry-loss-weight', '0.2',
        '--boundary-loss-weight', '0.1', '--depth-target-mode', 'projected',
        '--depth-loss-mode', 'bce', '--depth-loss-weight', '3',
        '--semantic-dice-weight', $diceWeight, '--bev-aux-weight', $bevWeight,
        '--dataset-version', 'v1.0-trainval', '--dataroot', 'data/occ3d_trainval',
        '--annotations-json', 'data/occ3d_trainval/downloads/annotations.json',
        '--occupancy-root', 'data/occ3d_trainval/gts',
        '--train-scenes', 'outputs/splits/train100_val20/train_scenes.json',
        '--val-scenes', 'outputs/splits/train100_val20/val_scenes.json',
        '--epochs', "$Epochs", '--overfit-samples', '0', '--class-balance', 'sqrt-inverse',
        '--batch-size', '1', '--workers', "$Workers", '--seed', "$Seed",
        '--output-dir', $experimentOutput
    )
    $experimentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $experimentPython)) { $experimentPython = 'python' }
    & $experimentPython @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
