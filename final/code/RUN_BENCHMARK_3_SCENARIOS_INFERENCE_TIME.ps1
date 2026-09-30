param(
  [string]$Python = "python",
  [string]$Device = "cuda:0",
  [int]$BatchSize = 16,
  [int]$NumWorkers = 0,
  [int]$WarmupBatches = 2,
  [int]$TimingRepeats = 1,
  [string]$OutputRoot = "final\Experiment\reports\inference_benchmark"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$cacheRoot = "final\data\Input\keypoint_tensor_cache"
$instanceGraphRoot = "final\data\Input\instance_graphs"
$templateDir = "final\data\Input\template_graphs"
$fusionTemplateDir = "data\template_graphs"
$testFile = "final\data\Input\test_1229.txt"
$neuralCheckpoint = "final\data\Output\checkpoints\neural_only_transformer_30epoch\last.pt"
$kgCheckpoint = "final\data\Output\checkpoints\kg_cleaned_3branch\last.pt"
$fusionCheckpoint = "final\data\Output\checkpoints\final_structured_req_epoch19\last.pt"

foreach ($requiredPath in @(
  $cacheRoot,
  $instanceGraphRoot,
  $templateDir,
  $fusionTemplateDir,
  $testFile,
  $neuralCheckpoint,
  $kgCheckpoint,
  $fusionCheckpoint
)) {
  if (-not (Test-Path -LiteralPath $requiredPath)) {
    throw "Required input not found: $requiredPath"
  }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$neuralOutput = Join-Path $OutputRoot "scenario_1_neural\metrics.json"
$kgOutput = Join-Path $OutputRoot "scenario_2_kg_guided\metrics.json"
$fusionOutput = Join-Path $OutputRoot "scenario_3_fusion\metrics.json"

function Invoke-TestCommand {
  param(
    [string]$Name,
    [string[]]$Arguments
  )

  Write-Host "Running $Name..." -ForegroundColor Cyan
  $previousErrorActionPreference = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  & $Python @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputRoot "$Name.log")
  $status = $LASTEXITCODE
  $ErrorActionPreference = $previousErrorActionPreference
  if ($status -ne 0) {
    throw "$Name failed (exit code $status)"
  }
}

$commonTimingArgs = @(
  "--batch-size", "$BatchSize",
  "--num-workers", "$NumWorkers",
  "--device", $Device,
  "--timing-warmup-batches", "$WarmupBatches",
  "--timing-repeats", "$TimingRepeats"
)

$neuralArgs = @(
  "final\code\neural_only_experiments\test_neural_only.py",
  "--checkpoint", $neuralCheckpoint,
  "--cache-root", $cacheRoot,
  "--test-file", $testFile,
  "--output", $neuralOutput
) + $commonTimingArgs
Invoke-TestCommand "scenario_1_neural" $neuralArgs

$crossAttentionScript = "final\code\src_neurosymbolic\test_cross_attention_v2.py"
$commonGraphArgs = @(
  "--keypoint-root", $cacheRoot,
  "--cache-root", $cacheRoot,
  "--instance-graph-root", $instanceGraphRoot,
  "--template-dir", $templateDir,
  "--test-file", $testFile
) + $commonTimingArgs

$kgArgs = @(
  $crossAttentionScript,
  "--checkpoint", $kgCheckpoint,
  "--eval-branch", "kg",
  "--output", $kgOutput
) + $commonGraphArgs
Invoke-TestCommand "scenario_2_kg_guided" $kgArgs

$fusionArgs = @(
  $crossAttentionScript,
  "--checkpoint", $fusionCheckpoint,
  "--keypoint-root", $cacheRoot,
  "--cache-root", $cacheRoot,
  "--instance-graph-root", $instanceGraphRoot,
  "--template-dir", $fusionTemplateDir,
  "--test-file", $testFile,
  "--eval-branch", "final",
  "--output", $fusionOutput
) + $commonTimingArgs
Invoke-TestCommand "scenario_3_fusion" $fusionArgs

$scenarioDefinitions = @(
  @{ Name = "Kich ban 1: Neural-only"; Path = $neuralOutput },
  @{ Name = "Kich ban 2: KG-guided"; Path = $kgOutput },
  @{ Name = "Kich ban 3: Neuro-Symbolic Fusion"; Path = $fusionOutput }
)

$summary = foreach ($scenario in $scenarioDefinitions) {
  $metrics = Get-Content -LiteralPath $scenario.Path -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    scenario = $scenario.Name
    accuracy = [double]$metrics.accuracy
    accuracy_percent = 100.0 * [double]$metrics.accuracy
    batch_size = [int]$metrics.inference_timing.batch_size
    timed_samples = [int]$metrics.inference_timing.timed_samples
    inference_seconds = [double]$metrics.inference_timing.total_seconds
    latency_ms_per_sample = [double]$metrics.inference_timing.latency_ms_per_sample
    throughput_samples_per_second = [double]$metrics.inference_timing.throughput_samples_per_second
    device = [string]$metrics.inference_timing.device
  }
}

$summaryPath = Join-Path $OutputRoot "inference_summary.csv"
$summary | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$summary | Format-Table scenario, accuracy_percent, latency_ms_per_sample, throughput_samples_per_second -AutoSize
Write-Host "Summary saved to $summaryPath" -ForegroundColor Green
