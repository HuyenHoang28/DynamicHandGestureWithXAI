param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$InputVideo,
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Device = "cpu",
  [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

function Resolve-PathFromProject {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return [System.IO.Path]::GetFullPath($Path)
  }

  return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Path))
}

function Require-File {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Description
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Description not found: $Path"
  }
}

function Require-Directory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [Parameter(Mandatory = $true)]
    [string]$Description
  )

  if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
    throw "$Description not found: $Path"
  }
}

$CallerDirectory = (Get-Location).Path
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Push-Location -LiteralPath $ProjectRoot
try {

if ([System.IO.Path]::IsPathRooted($InputVideo)) {
  $InputVideoPath = [System.IO.Path]::GetFullPath($InputVideo)
} else {
  $InputVideoPath = [System.IO.Path]::GetFullPath((Join-Path $CallerDirectory $InputVideo))
}

$PythonPath = Resolve-PathFromProject $Python
$DemoScriptPath = Join-Path $ProjectRoot "final\code\src_neurosymbolic\demo_video_v4.py"
$CheckpointPath = Join-Path $ProjectRoot "final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best_test.pt"
$TemplateDirPath = Join-Path $ProjectRoot "final\data\Input\template_graphs"
$ExtractorPath = Join-Path $ProjectRoot "keypoint_extractor\run_rtmw_splits_dual.py"
$MmposeRootPath = Join-Path $ProjectRoot ".venv\Lib\site-packages\mmpose\.mim"
$PoseWeightsPath = Join-Path $ProjectRoot "final\data\Output\checkpoints\rtmw\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  $OutputDirPath = Join-Path $ProjectRoot (Join-Path "final\outputs\demo_video_v4" ([System.IO.Path]::GetFileNameWithoutExtension($InputVideoPath)))
} else {
  $OutputDirPath = Resolve-PathFromProject $OutputDir
}

Require-File $InputVideoPath "Input video"
Require-File $PythonPath "Python executable"
Require-File $DemoScriptPath "V4 demo script"
Require-File $CheckpointPath "V4 fusion checkpoint"
Require-Directory $TemplateDirPath "Template graph directory"
Require-File $ExtractorPath "RTMW extractor script"
Require-Directory $MmposeRootPath "MMPose root"
Require-File $PoseWeightsPath "RTMW pose weights"

New-Item -ItemType Directory -Force -Path $OutputDirPath | Out-Null

# YAPF can hang while trying to use the restricted Windows LocalAppData cache.
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = Join-Path $ProjectRoot ".gitignore"
# The bundled OpenMMLab checkpoint is trusted and uses the legacy torch.load path.
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"

$DemoArguments = @(
  "--input-video", $InputVideoPath,
  "--checkpoint", $CheckpointPath,
  "--template-dir", $TemplateDirPath,
  "--output-dir", $OutputDirPath,
  "--device", $Device,
  "--repo-root", $ProjectRoot,
  "--extract-script", $ExtractorPath,
  "--mmpose-root", $MmposeRootPath,
  "--pose-weights", $PoseWeightsPath
)

Write-Host "Running V4 single-video demo on $Device`: $InputVideoPath"
Write-Host "Output directory: $OutputDirPath"
& $PythonPath $DemoScriptPath @DemoArguments
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
  throw "demo_video_v4.py failed with exit code $ExitCode."
}

}
finally {
  Pop-Location
}
