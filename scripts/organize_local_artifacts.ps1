$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

# Archive generated artifacts without deleting raw datasets, production inputs,
# environments, or checkpoints.
$moves = @(
    @('benchmark_data', 'results/experiments/date27/keypoint_benchmark_data'),
    @('final/outputs', 'results/experiments/legacy/final_outputs'),
    @('final/Experiment/reports', 'results/experiments/legacy/final_experiment_reports'),
    @('notebooks/IPN_Symbolic_V2_Colab_res.ipynb', 'results/experiments/ipn13/notebooks/IPN_Symbolic_V2_Colab_res.ipynb'),
    @('docs/research/SPIE_journal_papers__sample_manuscript_showing_style_and_formatting_specifications (1).pdf', 'docs/paper/DATE_NeuroSymbolic_Manuscript.pdf'),
    @('external_data/transformer_seed42-20261021T071234Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/transformer_seed42-20261021T071234Z-1-001.zip'),
    @('external_data/transformer_seed42-20261021T081614Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/transformer_seed42-20261021T081614Z-1-001.zip'),
    @('external_data/neural_ablation_B_masked_mean_matched_A-20261021T094533Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/neural_ablation_B_masked_mean_matched_A-20261021T094533Z-1-001.zip'),
    @('external_data/ipn_symbolic_v2_post_c-20261021T140256Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/ipn_symbolic_v2_post_c-20261021T140256Z-1-001.zip'),
    @('external_data/035179b821061ab7-20261022T015729Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/035179b821061ab7-20261022T015729Z-1-001.zip'),
    @('external_data/6efa8ad55dde2909-20261022T041458Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/6efa8ad55dde2909-20261022T041458Z-1-001.zip'),
    @('external_data/d255086f928a0615-20261022T075132Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/d255086f928a0615-20261022T075132Z-1-001.zip'),
    @('external_data/c46e0e469d2cf3b6-20261023T141655Z-1-001.zip', 'results/experiments/ipn13/artifact_zips/c46e0e469d2cf3b6-20261023T141655Z-1-001.zip')
)

function Resolve-WorkspacePath([string]$relative) {
    $absolute = [IO.Path]::GetFullPath((Join-Path $repoRoot $relative))
    if (-not $absolute.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside workspace: $absolute"
    }
    return $absolute
}

# Validate all exact endpoints before moving. Re-running after a successful move
# is safe; overwriting an existing destination is never allowed.
foreach ($pair in $moves) {
    $source = Resolve-WorkspacePath $pair[0]
    $destination = Resolve-WorkspacePath $pair[1]
    $hasSource = Test-Path -LiteralPath $source
    $hasDestination = Test-Path -LiteralPath $destination
    if (-not $hasSource -and -not $hasDestination) {
        Write-Warning "Skipped missing optional artifact: $($pair[0])"
    }
    if ($hasSource -and $hasDestination) {
        throw "Refusing to overwrite existing destination: $destination"
    }
}

foreach ($pair in $moves) {
    $source = Resolve-WorkspacePath $pair[0]
    $destination = Resolve-WorkspacePath $pair[1]
    if (-not (Test-Path -LiteralPath $source)) { continue }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Move-Item -LiteralPath $source -Destination $destination
    Write-Output "$($pair[0]) -> $($pair[1])"
}
