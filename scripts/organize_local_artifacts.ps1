$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$moves = @(
    @('external_data/ipn_full/videos/videos01.tgz', 'external_data/ipn_archives/videos01.tgz'),
    @('external_data/ipn_full/videos/videos02.tgz', 'external_data/ipn_archives/videos02.tgz'),
    @('external_data/ipn_full/videos/videos03.tgz', 'external_data/ipn_archives/videos03.tgz'),
    @('external_data/ipn_full/videos/videos04.tgz', 'external_data/ipn_archives/videos04.tgz'),
    @('external_data/ipn_full/videos/videos05.tgz', 'external_data/ipn_archives/videos05.tgz'),
    @('external_data/annotations-20261005T055143Z-1-001.zip', 'external_data/ipn_archives/annotations-20261005T055143Z-1-001.zip'),
    @('external_data/videos-20261005T055551Z-1-001.zip', 'external_data/ipn_archives/videos-20261005T055551Z-1-001.zip'),
    @('external_data/videos-20261005T055551Z-1-002.zip', 'external_data/ipn_archives/videos-20261005T055551Z-1-002.zip'),
    @('external_data/videos-20261005T055551Z-1-003.zip', 'external_data/ipn_archives/videos-20261005T055551Z-1-003.zip'),
    @('final/Experiment/reports', 'final/outputs/evaluations'),
    @('docs/benchmark_evidence', 'final/outputs/evidence/benchmark_evidence'),
    @('docs/demo_evidence', 'final/outputs/evidence/demo_evidence'),
    @('docs/keypoint_benchmark_evidence', 'final/outputs/evidence/keypoint_benchmark_evidence'),
    @('docs/model_evaluation_evidence', 'final/outputs/evidence/model_evaluation_evidence'),
    @('docs/deliverables', 'docs/reports/project'),
    @('docs/KEYPOINT_EXTRACTOR_BENCHMARK_REPORT.md', 'docs/reports/keypoint_benchmark/KEYPOINT_EXTRACTOR_BENCHMARK_REPORT.md'),
    @('docs/keypoint_benchmark_tradeoff.png', 'docs/reports/keypoint_benchmark/keypoint_benchmark_tradeoff.png'),
    @('final/outputs/ipn_smoke_20261005/REPORT_VI.md', 'docs/reports/ipn/REPORT_VI.md'),
    @('final/outputs/ipn_smoke_20261005/REPORT_VI.docx', 'docs/reports/ipn/REPORT_VI.docx'),
    @('final/outputs/ipn_smoke_20261005/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.md', 'docs/reports/ipn/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.md'),
    @('final/outputs/ipn_smoke_20261005/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.docx', 'docs/reports/ipn/BAO_CAO_KET_QUA_THU_NGHIEM_IPN_HAND.docx')
)
# Validate every source and destination before moving anything. Never overwrite.
foreach ($pair in $moves) {
    foreach ($relative in $pair) {
        $absolute = [IO.Path]::GetFullPath((Join-Path $repoRoot $relative))
        if (-not $absolute.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Path outside workspace: $absolute"
        }
    }
    $hasSource = Test-Path -LiteralPath (Join-Path $repoRoot $pair[0])
    $hasDestination = Test-Path -LiteralPath (Join-Path $repoRoot $pair[1])
    if (-not $hasSource -and -not $hasDestination) { throw "Missing source and destination: $($pair[0])" }
    if ($hasSource -and $hasDestination) { throw "Destination exists: $($pair[1])" }
}
foreach ($pair in $moves) {
    $source = Join-Path $repoRoot $pair[0]
    $destination = Join-Path $repoRoot $pair[1]
    if (-not (Test-Path -LiteralPath $source)) { continue }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    try {
        Move-Item -LiteralPath $source -Destination $destination
        Write-Output "$($pair[0]) -> $($pair[1])"
    } catch {
        Write-Warning "Not moved (close any application holding the file and rerun): $source. $($_.Exception.Message)"
    }
}
