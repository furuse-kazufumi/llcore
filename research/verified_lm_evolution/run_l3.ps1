# SPDX-License-Identifier: Apache-2.0
# Pre-registered R-LLM-0 L3 batch: mechanism (landscape) + payoff (gated real/null).
# Sequential so each run gets full CPU; logs + JSON results land in this dir.
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
# BLAS thread cap: avoids OpenBLAS oversubscription on tiny GEMMs (T x9 @ 9x256).
# Correctness-preserving (float64 + certifier margins untouched; SHA-256-stable output).
# Landscape (readout-dominated) clearly benefits; gated (cert-heavy) net effect to be measured.
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Set-Location -Path $PSScriptRoot
$py = 'py'
$args311 = '-3.11'

function Run($name, $script, $cliargs) {
    Write-Host "=== START $name @ $(Get-Date -Format o) ==="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $py $args311 $script @cliargs *>&1 | Tee-Object -FilePath "l3_$name.log"
    Write-Host "=== END   $name  elapsed=$([math]::Round($sw.Elapsed.TotalMinutes,1))min @ $(Get-Date -Format o) ==="
}

Run 'landscape' 'exp_landscape.py' @('900','12288')
Run 'gated_real' 'exp_gated.py'    @('15')
Run 'gated_null' 'exp_gated.py'    @('15','--null')
Write-Host "ALL_L3_RUNS_DONE @ $(Get-Date -Format o)"
