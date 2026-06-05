# SPDX-License-Identifier: Apache-2.0
# Gated-only L3 rerun (landscape already complete). NT=1 (measured win on the readout path;
# net effect on the cert-heavy gated path is measured here). exp_gated.py now checkpoints a
# partial-but-valid JSON after every seed, so this can be stopped early with data intact.
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Set-Location -Path $PSScriptRoot

function Run($name, $cliargs) {
    Write-Host "=== START $name @ $(Get-Date -Format o)  (NT=1) ==="
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & py -3.11 exp_gated.py @cliargs *>&1 | Tee-Object -FilePath "l3_$name.log"
    Write-Host "=== END   $name  elapsed=$([math]::Round($sw.Elapsed.TotalMinutes,1))min @ $(Get-Date -Format o) ==="
}

Run 'gated_real' @('15')
Run 'gated_null' @('15','--null')
Write-Host "ALL_GATED_RUNS_DONE @ $(Get-Date -Format o)"
