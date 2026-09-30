param(
  [string]$Python = "python",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 128,
  [int]$Epochs = 20
)

$ErrorActionPreference = "Stop"

$backbones = @("transformer", "tcn", "bigru", "mlp_mixer")
foreach ($backbone in $backbones) {
  $out = "outputs/neural_only_$backbone"
  & $Python neural_only_experiments\train_neural_only.py `
    --backbone $backbone `
    --cache-root data\keypoint_tensor_cache `
    --train-file train.txt `
    --val-file val.txt `
    --output-dir $out `
    --batch-size $BatchSize `
    --epochs $Epochs `
    --device $Device

  & $Python neural_only_experiments\test_neural_only.py `
    --checkpoint "$out\best.pt" `
    --cache-root data\keypoint_tensor_cache `
    --test-file test.txt `
    --batch-size $BatchSize `
    --device $Device `
    --output "$out\test_metrics.json"
}
