# SPDX-License-Identifier: Apache-2.0
# L3 follow-up batch (resolves red-team residuals):
#  (1) 8192B landscape on the SAME corpus as the gated run -> resolve ceiling-vs-navigability
#      (does the inf region CONTAIN better-than-unigram genes on 8192B that inf-gated evolution
#       failed to reach? if yes => the inf handicap is partly navigability, not pure ceiling).
#  (2) null control (exp_gated.py --null): shuffle the corpus to destroy sequential structure ->
#      all gates MUST tie (pre-reg gate L3-null, the load-bearing falsifier).
# exp_*.py checkpoint after each unit, so this is kill-safe. NT=1 (landscape is readout-dominated).
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Set-Location -Path $PSScriptRoot

# Preserve the canonical 12288B / 900-gene mechanism result before the 8192B run overwrites it.
if (Test-Path exp_landscape_results.json) {
  Copy-Item exp_landscape_results.json exp_landscape_12288_results.json -Force
  Write-Host "backed up 12288B landscape -> exp_landscape_12288_results.json"
}

Write-Host "=== START landscape_8192 @ $(Get-Date -Format o)  (NT=1) ==="
$sw = [Diagnostics.Stopwatch]::StartNew()
& py -3.11 exp_landscape.py 500 8192 *>&1 | Tee-Object -FilePath "l3_landscape_8192.log"
if (Test-Path exp_landscape_results.json) { Copy-Item exp_landscape_results.json exp_landscape_8192_results.json -Force }
Write-Host "=== END landscape_8192 elapsed=$([math]::Round($sw.Elapsed.TotalMinutes,1))min ==="

Write-Host "=== START gated_null @ $(Get-Date -Format o)  (NT=1) ==="
$sw2 = [Diagnostics.Stopwatch]::StartNew()
& py -3.11 exp_gated.py 15 --null *>&1 | Tee-Object -FilePath "l3_gated_null.log"
Write-Host "=== END gated_null elapsed=$([math]::Round($sw2.Elapsed.TotalMinutes,1))min ==="
Write-Host "ALL_L3B_DONE @ $(Get-Date -Format o)"
