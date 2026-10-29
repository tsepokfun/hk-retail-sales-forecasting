# Driver for the Stage-1 exogenous ablation.
#
# One python process per fit.  This box has 32 GB of RAM, already dipped below
# 5 GB free, and once died with "The paging file is too small" when a second
# torch process loaded -- so the fits must not overlap.  A fresh interpreter per
# fit also gives each fit a clean CUDA context, which the earlier
# "CUDA error: out of memory" (with 11 GB actually free) turned out to need.
#
# Resumable: run_exog_ablation.py --only <cell>:<seed> skips a finished tag.
#
# Usage
#   pwsh -NoProfile -File run_exog_all.ps1
#   pwsh -NoProfile -File run_exog_all.ps1 -Epochs 100 -Cells none,macro_fin

param(
    [int]$Epochs = 100,
    [int]$Retries = 6,
    [int]$RetrySleepSec = 60,
    # NOTE: keep this a plain [string].  Start-Process -ArgumentList and
    # `pwsh -File` pass raw tokens, so `-Cells a,b,c` arrives as ONE string and a
    # [string[]] parameter binds it as a single element (which silently produced
    # a cell literally named "a,b,c").  We split it ourselves below.
    [string]$Cells = 'none,macro_fin,people_weather_logistics,all,random',
    [string]$Seeds = '42,7,2024',
    [switch]$Raw
)

$CellList = @($Cells -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$SeedList = @($Seeds -split ',' | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ })

$ErrorActionPreference = 'Continue'
$wd = $PSScriptRoot
Set-Location $wd

# ---- single-instance lock -------------------------------------------------
$lock = Join-Path $wd 'results\exog_driver.lock'
if (Test-Path $lock) {
    $old = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "driver already running as PID $old -- refusing to start a second one"
        exit 3
    }
    Write-Host "removing stale lock (PID $old is gone)"
}
Set-Content -Path $lock -Value $PID -Encoding UTF8
trap { Remove-Item $lock -ErrorAction SilentlyContinue; break }

$py   = Join-Path $wd '.venv\Scripts\python.exe'
$log  = Join-Path $wd 'results\exog_ablation.log'
$errl = Join-Path $wd 'results\exog_ablation.err.log'

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Log "============ EXOG DRIVER START (epochs=$Epochs, cells=$($CellList -join ','), seeds=$($SeedList -join ',')) ============"
$t0 = Get-Date
$nfail = 0

foreach ($cell in $CellList) {
    foreach ($seed in $SeedList) {
        # MUST match run_exog_ablation.set_mode(): the default (non -Raw) tree is
        # exog_preds_z.  Pointing this at exog_preds would make every tag look
        # missing and burn 6 retries per tag.
        $sub = if ($Raw) { 'exog_preds' } else { 'exog_preds_z' }
        $csv = Join-Path $wd ("results\{2}\xLSTM_{0}_s{1}.csv" -f $cell, $seed, $sub)
        if (Test-Path $csv) { Log "[skip] ${cell}:$seed already done"; continue }

        $ok = $false
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            Log "[run ] ${cell}:$seed  attempt $attempt/$Retries"
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $rawFlag = if ($Raw) { @('--raw') } else { @() }
            & $py -u run_exog_ablation.py --only "${cell}:$seed" --epochs $Epochs @rawFlag *>> $log 2>> $errl
            $code = $LASTEXITCODE
            $sw.Stop()
            Log ("[done] {0}:{1} exit={2} in {3:N0}s" -f $cell, $seed, $code, $sw.Elapsed.TotalSeconds)

            if ((Test-Path $csv) -and $code -eq 0) { $ok = $true; break }
            if ($code -ne 0) {
                Log "[fail] ${cell}:$seed exit=$code -- sleeping ${RetrySleepSec}s"
                Start-Sleep -Seconds $RetrySleepSec
            }
        }
        if (-not $ok) { Log "[FAIL] ${cell}:$seed gave up after $Retries attempts"; $nfail++ }

        # be polite to a nearly-full page file
        [System.GC]::Collect()
        Start-Sleep -Seconds 5
    }
}

Log "================ COMBINE ================  (failures so far: $nfail)"
$combineFlag = if ($Raw) { @('--raw') } else { @() }
& $py -u run_exog_ablation.py --combine-only @combineFlag *>> $log 2>> $errl
Log "[combine] exit=$LASTEXITCODE"

$dt = (Get-Date) - $t0
Log ("============ EXOG DRIVER END in {0:N1} min, failures {1} ============" -f $dt.TotalMinutes, $nfail)
Remove-Item $lock -ErrorAction SilentlyContinue
