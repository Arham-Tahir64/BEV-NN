param(
    [ValidateRange(1,10000)][int]$Epochs=8,
    [ValidateSet(320,480)][int]$ImageHeight=320,
    [ValidateRange(0,2147483647)][int]$Seed=0,
    [ValidateRange(0,64)][int]$Workers=2,
    [ValidateRange(0,100)][double]$DepthWeight=0,
    [ValidateRange(1,1000000)][int]$QueryChunk=4096,
    [switch]$LearnedOffsets,
    [string]$OutputDir=''
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Push-Location $projectRoot
try {
    $imageWidth=if($ImageHeight -eq 320){576}else{864}
    $depthName=$DepthWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    if(-not $OutputDir){$OutputDir="foundation4d/outputs/lite_${ImageHeight}_depth${depthName}_offset$([int]$LearnedOffsets.IsPresent)_seed${Seed}"}
    $experimentPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
    if(-not(Test-Path -LiteralPath $experimentPython)){$experimentPython='python'}
    $trainingArgs=@('-u','-m','foundation4d.train','--project-root',$projectRoot,
        '--epochs',"$Epochs",'--image-size',"$ImageHeight","$imageWidth",'--seed',"$Seed",
        '--workers',"$Workers",'--depth-weight',$depthName,'--query-chunk',"$QueryChunk",'--output-dir',$OutputDir)
    if($LearnedOffsets){$trainingArgs+='--learned-offsets'}
    & $experimentPython @trainingArgs
    if($LASTEXITCODE -ne 0){throw "Training exited with code $LASTEXITCODE"}
} finally {Pop-Location}
