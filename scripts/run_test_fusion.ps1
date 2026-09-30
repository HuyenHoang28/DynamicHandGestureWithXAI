param(
  [string]$Python = ".\.venv\Scripts\python.exe",
  [string]$Device = "cuda:0",
  [string]$Checkpoint = "final\data\Output\checkpoints\cross_attention_v4_template170_retrain_next\best_test.pt",
  [string]$Output = "final\Experiment\reports\packaged_test_fusion_best_test"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

& $Python final\code\src_neurosymbolic\test_cross_attention_v2.py `
  --checkpoint $Checkpoint `
  --keypoint-root final\data\Input\keypoint_tensor_cache `
  --cache-root final\data\Input\keypoint_tensor_cache `
  --instance-graph-root final\data\Input\instance_graphs `
  --template-dir final\data\Input\template_graphs `
  --test-file final\data\Input\test_1229.txt `
  --batch-size 16 `
  --num-workers 0 `
  --eval-branch final `
  --device $Device `
  --output $Output
