# Stage 0 -- modern-baseline campaign driver.
#
# STRICTLY SEQUENTIAL, ONE FIT PER PROCESS.  Every entry below launches a fresh
# interpreter and WAITS for it to exit before the next one starts, because this
# box has produced a phantom `CUDA error: out of memory` when two torch processes
# coexisted.  There is no `Start-Job`, no `ForEach-Object -Parallel`, no `&`.
#
# Re-running this script is safe: run_sota_baselines.py skips a tag whose
# prediction CSV already exists, so an interrupted campaign resumes where it
# stopped.
#
# Order = the brief's priority order.  iTransformer (ablation #9) first, then the
# linear models, then the anchor-matched and cost control arms last, because the
# last arms are the ones that get cut if the GPU budget runs out.
#
# Kernel choice for DLinear: k=5 was selected on the PURGED validation block
# (targets <= 202312) at seed 42 (k=3 val 0.4502 vs k=5 val 0.4461).  k=3 is
# still run at every seed as a sensitivity check, never as the headline.

$ErrorActionPreference = "Continue"
$py = ".venv\Scripts\python.exe"
$log = "results\sota_campaign.log"
$fail = 0
$runs = 0

function Run-Fit([string]$family, [int]$seed, [int]$kernel) {
    $tag = "$family`_s$seed" + $(if ($family -like "*dlinear*") { "_k$kernel" } else { "" })
    Write-Output ("=" * 78) | Tee-Object -FilePath $log -Append
    Write-Output ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $tag) | Tee-Object -FilePath $log -Append
    Write-Output ("=" * 78) | Tee-Object -FilePath $log -Append
    & $py run_sota_baselines.py --family $family --seed $seed --kernel $kernel --epochs 100 2>&1 |
        Tee-Object -FilePath $log -Append |
        Select-String -Pattern ">>>|ERROR|Traceback|RuntimeError|CUDA" | Out-String -Width 160
    $script:runs++
    if ($LASTEXITCODE -ne 0) {
        $script:fail++
        Write-Output ("  !! exit code {0} for {1}" -f $LASTEXITCODE, $tag) | Tee-Object -FilePath $log -Append
    }
}

$t0 = Get-Date
Write-Output "campaign start $t0" | Tee-Object -FilePath $log

$K = 5

# ---- 1. ablation #9: channel mixing vs channel independence (5 seeds) -------
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "itransformer" $s 0 }
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "timetf" $s 0 }

# ---- 2. DLinear / NLinear, shared weights (5 seeds, k=5 + k=3 sensitivity) --
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "dlinear_shared" $s $K }
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "dlinear_shared" $s 3 }
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "nlinear_shared" $s 0 }

# ---- 3. DLinear / NLinear, 23 independent fits per seed (3 seeds) ----------
foreach ($s in 42, 7, 2024) { Run-Fit "dlinear_perseries" $s $K }
foreach ($s in 42, 7, 2024) { Run-Fit "nlinear_perseries" $s 0 }

# ---- 4. anchor-matched arms (the champion's learnable 2-term anchor) -------
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "itransformer_anchor" $s 0 }
foreach ($s in 42, 7, 2024, 1, 1234) { Run-Fit "timetf_anchor" $s 0 }

# ---- 5. control: the xLSTM architecture with the persistence anchor --------
foreach ($s in 42, 7, 2024) { Run-Fit "xlstm_persist" $s 0 }

$dt = (Get-Date) - $t0
Write-Output ("campaign end: {0} runs, {1} failures, elapsed {2:hh\:mm\:ss}" -f $runs, $fail, $dt) |
    Tee-Object -FilePath $log -Append
