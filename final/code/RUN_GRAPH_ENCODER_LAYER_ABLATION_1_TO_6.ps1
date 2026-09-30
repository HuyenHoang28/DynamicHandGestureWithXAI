param(
  [string]$Python = "python",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 16,
  [int]$Epochs = 30,
  [int]$NumWorkers = 0,
  [int]$Seed = 42,
  [int]$EarlyStopPatience = 5,
  [int]$StartGraphLayers = 1,
  [int]$EndGraphLayers = 6,
  [string]$KeypointRoot = "final\data\Input\keypoint_tensor_cache",
  [string]$CacheRoot = "final\data\Input\keypoint_tensor_cache",
  [string]$InstanceGraphRoot = "final\data\Input\instance_graphs",
  [string]$TemplateDir = "final\data\Input\template_graphs",
  [string]$TrainFile = "final\data\Input\train.txt",
  [string]$ValFile = "final\data\Input\val.txt",
  [string]$OutputRoot = "final\outputs\graph_encoder_layer_ablation_scratch"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

foreach ($requiredPath in @(
  $KeypointRoot,
  $CacheRoot,
  $InstanceGraphRoot,
  $TemplateDir,
  $TrainFile,
  $ValFile
)) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required input not found: $requiredPath"
  }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$trainScript = "final\code\src_neurosymbolic\train_cross_attention_v4.py"
$summaryRows = @()
$summaryPath = Join-Path $OutputRoot "summary.csv"
if (Test-Path -LiteralPath $summaryPath) {
  $summaryRows = @(Import-Csv -LiteralPath $summaryPath)
}

if ($StartGraphLayers -lt 1 -or $EndGraphLayers -lt $StartGraphLayers) {
  throw "Invalid graph layer range: $StartGraphLayers..$EndGraphLayers"
}

foreach ($graphLayers in $StartGraphLayers..$EndGraphLayers) {
  $outputDir = Join-Path $OutputRoot ("graph_layers_{0:D2}" -f $graphLayers)
  New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
  $logFile = Join-Path $outputDir "train.log"
  $startedAt = Get-Date

  Write-Host "Training from scratch with Graph Encoder layers=$graphLayers..." -ForegroundColor Cyan
  $argsList = @(
    "-u", $trainScript,
    "--keypoint-root", $KeypointRoot,
    "--cache-root", $CacheRoot,
    "--instance-graph-root", $InstanceGraphRoot,
    "--template-dir", $TemplateDir,
    "--train-file", $TrainFile,
    "--val-file", $ValFile,
    "--output-dir", $outputDir,
    "--logit-branch", "kg",
    "--batch-size", "$BatchSize",
    "--epochs", "$Epochs",
    "--layers", "3",
    "--graph-layers", "$graphLayers",
    "--decoder-layers", "1",
    "--instance-tokens", "16",
    "--dropout", "0.15",
    "--neural-lr", "0.0002",
    "--kg-lr", "0.0003",
    "--weight-decay", "0.01",
    "--early-stop-patience", "$EarlyStopPatience",
    "--num-workers", "$NumWorkers",
    "--seed", "$Seed",
    "--device", $Device,
    "--directed-temporal-edges"
  )

  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $Python @argsList 2>&1 | Tee-Object -FilePath $logFile
  $status = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorActionPreference
  if ($status -ne 0) {
    throw "Training failed for graph_layers=$graphLayers (exit code $status)"
  }

  $historyPath = Join-Path $outputDir "history.json"
  $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
  $best = $history | Sort-Object { [double]$_.val.kg_accuracy } -Descending | Select-Object -First 1
  $elapsedMinutes = ((Get-Date) - $startedAt).TotalMinutes

  $summaryRows = @($summaryRows | Where-Object { [int]$_.graph_layers -ne $graphLayers })
  $summaryRows += [PSCustomObject]@{
    graph_layers = $graphLayers
    temporal_layers = 3
    decoder_layers = 1
    seed = $Seed
    best_epoch = [int]$best.epoch
    train_kg_accuracy = [double]$best.train.accuracy
    val_kg_accuracy = [double]$best.val.kg_accuracy
    val_final_accuracy = [double]$best.val.final_accuracy
    val_neural_accuracy = [double]$best.val.neural_accuracy
    val_loss = [double]$best.val.loss
    epochs_ran = @($history).Count
    training_minutes = $elapsedMinutes
    output_dir = $outputDir
  }
  $summaryRows |
    Sort-Object { [int]$_.graph_layers } |
    Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
}

$summaryRows |
  Sort-Object val_kg_accuracy -Descending |
  Format-Table graph_layers, best_epoch, val_kg_accuracy, val_loss, epochs_ran, training_minutes -AutoSize
Write-Host "Summary saved to $summaryPath" -ForegroundColor Green
