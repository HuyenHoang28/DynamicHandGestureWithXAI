param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Device = "cuda:0",
  [string]$Checkpoint = "final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best.pt",
  [string]$Output = "results\runs\date27_v4_validation_selected",
  [switch]$AllowTestSelected
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
function Resolve-ProjectPath([string]$PathValue) {
  if ([IO.Path]::IsPathRooted($PathValue)) {
    return [IO.Path]::GetFullPath($PathValue)
  }
  return [IO.Path]::GetFullPath((Join-Path $ProjectRoot $PathValue))
}
$CheckpointPath = Resolve-ProjectPath $Checkpoint
$TestSelectedPath = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best_test.pt"))
if ($CheckpointPath -eq $TestSelectedPath -and -not $AllowTestSelected) {
  throw "best_test.pt was selected using test accuracy. Pass -AllowTestSelected only for explicitly exploratory evaluation."
}
$OutputPath = if ([IO.Path]::GetExtension($Output)) {
  $Output
} else {
  Join-Path $Output "test_metrics.json"
}

& $Python final\code\src_neurosymbolic\test_cross_attention_v2.py `
  --checkpoint $CheckpointPath `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --test-file final\data\Input\test_1229.txt `
  --batch-size 16 `
  --num-workers 0 `
  --eval-branch final `
  --device $Device `
  --output $OutputPath

if ($LASTEXITCODE -ne 0) {
  throw "Fusion evaluation failed with exit code $LASTEXITCODE"
}
