[CmdletBinding()]
param(
    [string]$DataRoot = (Join-Path $PSScriptRoot '..\external_data\ipn_processed\clips'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\external_data\ipn_processed\keypoints'),
    [string]$MmposeRoot = (Join-Path $PSScriptRoot '..\.venv\Lib\site-packages\mmpose\.mim'),
    [string]$PoseWeights = (Join-Path $PSScriptRoot '..\final\data\Output\checkpoints\rtmw\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth'),
    [ValidateSet('cuda:0', 'cpu')]
    [string]$Device = 'cuda:0',
    [switch]$Overwrite,
    [switch]$NoCache
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$dataRootPath = (Resolve-Path $DataRoot).Path
$outputRootPath = if ([IO.Path]::IsPathRooted($OutputRoot)) {
    [IO.Path]::GetFullPath($OutputRoot)
} else {
    [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputRoot))
}
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$runner = Join-Path $repoRoot 'keypoint_extractor\run_rtmw_splits_dual.py'
$logDir = Join-Path $repoRoot 'logs'
$logPath = Join-Path $logDir ('keypoint_rerun_{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

if (-not (Test-Path -LiteralPath $python)) { throw "Python environment not found: $python" }
if (-not (Test-Path -LiteralPath $runner)) { throw "Runner not found: $runner" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($split in @('train', 'val', 'test')) {
    $splitData = Join-Path $dataRootPath $split
    if (-not (Test-Path -LiteralPath $splitData)) { continue }
    $splitOutput = Join-Path $outputRootPath $split
    $args = @(
        $runner, '--repo-root', $repoRoot, '--data-root', $splitData,
        '--layout', 'raw-subjects', '--output-root', $splitOutput,
        '--mmpose-root', $MmposeRoot, '--pose-weights', $PoseWeights,
        '--device', $Device, '--pretty'
    )
    if ($Overwrite) { $args += '--overwrite' }
    if ($NoCache) { $args += '--no-cache' }
    Write-Host "Processing $split. Log: $logPath"
    & $python @args 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "Runner failed for $split with exit code $LASTEXITCODE" }
}

$missing = @()
foreach ($split in @('train', 'val', 'test')) {
    $splitData = Join-Path $dataRootPath $split
    $splitOutput = Join-Path $outputRootPath $split
    if (-not (Test-Path -LiteralPath $splitData)) { continue }
    Get-ChildItem -LiteralPath $splitData -Directory | ForEach-Object {
        $label = $_.Name
        Get-ChildItem -LiteralPath $_.FullName -Directory | ForEach-Object {
            $subject = $_.Name
            Get-ChildItem -LiteralPath $_.FullName -File |
                Where-Object { $_.Extension -in '.mp4', '.avi', '.mov', '.mkv' } |
                ForEach-Object {
                    $expected = Join-Path (Join-Path (Join-Path $splitOutput $label) $subject) ($_.BaseName + '.json')
                    if (-not (Test-Path -LiteralPath $expected)) {
                        $missing += "$split/$label/$subject/$($_.Name) -> $expected"
                    }
                }
        }
    }
}
if ($missing.Count -gt 0) {
    Write-Error ("{0} input videos still have no matching JSON:`n{1}" -f $missing.Count, ($missing -join "`n"))
    exit 2
}
Write-Host 'OK: every input video has a matching JSON.' -ForegroundColor Green
