param(
  [string]$Python = "python",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 16,
  [int]$Epochs = 30,
  [int]$NumWorkers = 0,
  [int]$Seed = 42,
  [int]$EarlyStopPatience = 5,
  [string]$CacheRoot = "final\data\Input\keypoint_tensor_cache",
  [string]$TrainFile = "final\data\Input\train.txt",
  [string]$ValFile = "final\data\Input\val.txt",
  [string]$OutputRoot = "final\outputs\neural_only_layer_ablation"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $ProjectRoot

foreach ($requiredPath in @($CacheRoot, $TrainFile, $ValFile)) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required input not found: $requiredPath"
  }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$trainScript = "final\code\neural_only_experiments\train_neural_only.py"
$summaryRows = @()

foreach ($layers in 1..10) {
  $outputDir = Join-Path $OutputRoot ("layers_{0:D2}" -f $layers)
  New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
  $logFile = Join-Path $outputDir "train.log"

  Write-Host "Training Transformer with $layers layer(s)..." -ForegroundColor Cyan
  $argsList = @(
    "-u", $trainScript,
    "--backbone", "transformer",
    "--cache-root", $CacheRoot,
    "--train-file", $TrainFile,
    "--val-file", $ValFile,
    "--output-dir", $outputDir,
    "--batch-size", "$BatchSize",
    "--epochs", "$Epochs",
    "--layers", "$layers",
    "--num-workers", "$NumWorkers",
    "--seed", "$Seed",
    "--early-stop-patience", "$EarlyStopPatience",
    "--device", $Device
  )

  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $Python @argsList 2>&1 | Tee-Object -FilePath $logFile
  $status = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorActionPreference
  if ($status -ne 0) {
    throw "Training failed for layers=$layers (exit code $status)"
  }

  $historyPath = Join-Path $outputDir "history.json"
  $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
  $best = $history | Sort-Object { [double]$_.val.accuracy } -Descending | Select-Object -First 1
  $summaryRows += [PSCustomObject]@{
    layers = $layers
    seed = $Seed
    best_epoch = [int]$best.epoch
    train_accuracy = [double]$best.train.accuracy
    val_accuracy = [double]$best.val.accuracy
    train_loss = [double]$best.train.loss
    val_loss = [double]$best.val.loss
    epochs_ran = @($history).Count
    output_dir = $outputDir
  }
  $summaryRows | Export-Csv -LiteralPath (Join-Path $OutputRoot "summary.csv") -NoTypeInformation -Encoding UTF8
}

$summaryRows | Sort-Object val_accuracy -Descending | Format-Table -AutoSize
Write-Host "Summary saved to $OutputRoot\summary.csv" -ForegroundColor Green
