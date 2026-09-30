param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Device = "cuda:0",
  [int]$Epochs = 30,
  [int]$BatchSize = 32,
  [string]$OutputDir = "final\outputs\train_v4_fusion_rerun"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& $Python final\code\src_neurosymbolic\train_cross_attention_v4.py `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --train-file final\data\Input\train.txt `
  --val-file final\data\Input\val.txt `
  --output-dir $OutputDir `
  --logit-branch final `
  --batch-size $BatchSize `
  --epochs $Epochs `
  --device $Device
