# Stage H GPU campaign driver.
#
#   pwsh -File run_stage_h.ps1 -Specs sw:base:1 hz:visitor_mtr_h12:42
#
# Spec format:  sw:<config>:<seed>   -> run_stage_h_sweep.py   (task A/B)
#               hz:<cell>:<seed>     -> run_stage_h_horizon.py (task C)
#
# HARD RULES ENFORCED HERE
#   * strictly sequential: this loop is the only thing allowed to start a torch
#     process (two coexisting torch processes have produced a phantom CUDA OOM
#     on this box), so a fit is never launched until the previous one exited;
#   * every fit is a FRESH interpreter;
#   * a non-zero exit is retried up to 3 times, and the full error text of each
#     attempt is kept in results/stage_h_logs/<tag>.log -- nothing is skipped
#     silently;
#   * per-fit wall time is appended to results/stage_h_campaign.log for the
#     cost accounting section of the report.
param([string[]]$Specs = @())

$ErrorActionPreference = 'Continue'
# `pwsh -File ... -Specs a,b,c` binds the WHOLE comma list as ONE string, so
# expand it here; otherwise "sw:base:1,sw:base:1234" becomes a single bogus spec.
$Specs = @($Specs | ForEach-Object { $_ -split ',' } | Where-Object { $_ -ne '' })
if ($Specs.Count -eq 0) { throw "no specs given" }
$py = ".\.venv\Scripts\python.exe"
$logdir = "results\stage_h_logs"
$campaign = "results\stage_h_campaign.log"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

foreach ($s in $Specs) {
    $parts = $s.Split(":")
    $kind = $parts[0]; $name = $parts[1]; $seed = $parts[2]
    $tag = "${kind}__${name}__s${seed}"
    $log = Join-Path $logdir "$tag.log"

    $ok = $false
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        $t0 = Get-Date
        if ($kind -eq "sw") {
            & $py run_stage_h_sweep.py --config $name --seed $seed *> $log
        } elseif ($kind -eq "hz") {
            & $py run_stage_h_horizon.py --cell $name --seed $seed *> $log
        } else {
            Write-Host "  unknown spec kind '$kind' in '$s'" -ForegroundColor Red
            break
        }
        $rc = $LASTEXITCODE
        $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        $verdict = "attempt=$attempt rc=$rc wall=${sec}s"
        if ($rc -eq 0) {
            $ok = $true
            $line = "$(Get-Date -Format o)`t$s`t$verdict`tOK"
            Add-Content -Path $campaign -Value $line
            Write-Host $line
            break
        }
        $line = "$(Get-Date -Format o)`t$s`t$verdict`tFAIL"
        Add-Content -Path $campaign -Value $line
        Write-Host $line -ForegroundColor Yellow
        if ($attempt -lt 3) {
            Write-Host "  retrying after 20s (checking no other torch process)..." -ForegroundColor Yellow
            Start-Sleep -Seconds 20
        }
    }
    if (-not $ok) {
        Add-Content -Path $campaign -Value "$(Get-Date -Format o)`t$s`tGAVE-UP after 3 attempts"
        Write-Host "  GAVE UP on $s -- see $log" -ForegroundColor Red
    }
}
Write-Host "campaign segment done: $($Specs.Count) spec(s)"
Get-Content $campaign | Select-Object -Last 12
