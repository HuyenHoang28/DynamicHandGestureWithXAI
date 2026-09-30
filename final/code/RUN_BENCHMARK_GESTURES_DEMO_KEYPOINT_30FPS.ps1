param(
  [string]$Python = "python",
  [string]$Device = "cuda:0",
  [int]$WarmupFrames = 10,
  [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$argsList = @(
  "-u",
  "final\code\benchmark_rtmw_folder_extraction_30fps.py",
  "--video-dir", "gestures_demo_mp4",
  "--device", $Device,
  "--warmup-frames", "$WarmupFrames"
)
if ($Limit -gt 0) {
  $argsList += @("--limit", "$Limit")
}

$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python @argsList
$status = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
exit $status
