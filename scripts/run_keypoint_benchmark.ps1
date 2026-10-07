param(
    [string]$VideoRoot = "benchmark_data\raw\gestures_demo_mp4",
    [string]$OutputDir = "final\outputs\evaluations\keypoint_model_benchmark",
    [string]$ModelsJson = "final\code\keypoint_benchmark_models.json",
    [string]$Device = "cpu",
    [int]$FramesPerVideo = 8,
    [double]$ConfidenceThreshold = 0.3,
    [Parameter(Mandatory = $true)]
    [ValidateSet("rtmw_m_256x192", "rtmw_l_384x288")]
    [string]$ModelName,
    [int]$FullKeypointsClasses = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$env:WIN_PD_OVERRIDE_LOCAL_APPDATA = (Resolve-Path (Join-Path $repoRoot ".gitignore")).Path
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"

Push-Location $repoRoot
try {
    $benchmarkArgs = @(
        "final\code\benchmark_keypoint_models.py",
        "--video-root", $VideoRoot,
        "--models-json", $ModelsJson,
        "--output-dir", $OutputDir,
        "--device", $Device,
        "--frames-per-video", $FramesPerVideo,
        "--confidence-threshold", $ConfidenceThreshold
    )
    if ($ModelName) { $benchmarkArgs += @("--model-name", $ModelName) }
    if ($FullKeypointsClasses -gt 0) {
        $benchmarkArgs += @("--full-keypoints-classes", $FullKeypointsClasses)
    }
    & $python @benchmarkArgs
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
