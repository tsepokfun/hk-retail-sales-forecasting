# Re-run the arms affected by the per-series anchor bug, plus the trainable-kernel
# control.  Strictly sequential, one fit per process, same as run_sota.ps1.
#
# WHAT WAS BROKEN: `neural_linear.fit_per_series` trained its 23 small models
# against RESIDUAL targets (y_z(t+h) - y_z(t)) while `DLinearNet.forward` already
# returns `anchor + delta`, and `run_sota_baselines` then added the anchor a
# second time.  The bug was caught by comparing against the shared-weights arm of
# the same model (MASE 2.886 vs 1.022) and by y_pred exceeding the panel maximum
# (21,010 vs max y_true 10,323).  Fixed in models/neural_linear.py (targets are
# now levels, one convention everywhere) AND in run_sota_baselines.py (no second
# anchor).  This script deletes the three poisoned DLinear per-series files and
# refits them with the fixed code; the NLinear per-series runs already used the
# fixed code and are kept.

$ErrorActionPreference = "Continue"
$py = ".venv\Scripts\python.exe"
$log = "results\sota_fix.log"
$fail = 0
$runs = 0

function Run-Fit([string]$family, [int]$seed, [int]$kernel) {
    $tag = "$family`_s$seed" + $(if ($family -like "*dlinear*") { "_k$kernel" } else { "" })
    Write-Output ("=" * 78) | Tee-Object -FilePath $log -Append
    Write-Output ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $tag) | Tee-Object -FilePath $log -Append
    & $py run_sota_baselines.py --family $family --seed $seed --kernel $kernel --epochs 100 2>&1 |
        Tee-Object -FilePath $log -Append |
        Select-String -Pattern ">>>|Traceback|RuntimeError|CUDA" | Out-String -Width 160
    $script:runs++
    if ($LASTEXITCODE -ne 0) { $script:fail++; Write-Output "  !! exit $LASTEXITCODE for $tag" | Tee-Object -FilePath $log -Append }
}

$t0 = Get-Date
Write-Output "fix campaign start $t0" | Tee-Object -FilePath $log

# ---- 1. delete the poisoned DLinear per-series outputs ---------------------
foreach ($s in 42, 7, 2024) {
    foreach ($p in @("results\sota_preds\dlinear_perseries_s$s`_k5.csv",
                     "results\sota_meta\dlinear_perseries_s$s`_k5.json")) {
        if (Test-Path $p) {
            Remove-Item $p
            Write-Output "  removed $p" | Tee-Object -FilePath $log -Append
        }
    }
}

# ---- 2. refit them with the fixed code -------------------------------------
foreach ($s in 42, 7, 2024) { Run-Fit "dlinear_perseries" $s 5 }

# ---- 3. the trainable-kernel control (brief's wording vs the paper) --------
foreach ($s in 42, 7, 2024) { Run-Fit "dlinear_kt" $s 5 }

# ---- 4. the sharpest #9 control: same iTransformer, attention bypassed ------
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "itransformer_noattn" $s 0 }

$dt = (Get-Date) - $t0
Write-Output ("fix campaign end: {0} runs, {1} failures, elapsed {2:hh\:mm\:ss}" -f $runs, $fail, $dt) |
    Tee-Object -FilePath $log -Append
