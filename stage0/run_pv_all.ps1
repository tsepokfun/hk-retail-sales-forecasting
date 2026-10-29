# Driver for the leaky-vs-purged validation comparison.
#
# One python process per fit, so each fit gets a fresh CUDA context.  This box
# has produced transient "CUDA error: out of memory" / "unknown error" when
# several fits share a single process, and a fresh context costs only a couple
# of seconds because the model is 20k parameters.
#
# Resumable: run_purged_val.py --only <proto>:<seed> skips a tag whose
# results/pv_preds/<tag>.csv already exists, so re-running this script is safe.
#
# Usage
#   powershell -ExecutionPolicy Bypass -File run_pv_all.ps1
#   powershell -ExecutionPolicy Bypass -File run_pv_all.ps1 -Epochs 100 -Retries 3

param(
    [int]$Epochs = 100,
    [int]$Retries = 6,
    [int]$RetrySleepSec = 45
)

$ErrorActionPreference = 'Continue'
$wd = $PSScriptRoot
Set-Location $wd

# ---- single-instance lock -------------------------------------------------
# Two drivers running at once means two CUDA contexts competing for one GPU,
# which on this box surfaces as a bogus "CUDA error: out of memory" with 11 GB
# actually free.  Refuse to start a second copy.
$lock = Join-Path $wd 'results\pv_driver.lock'
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
$log  = Join-Path $wd 'results\purged_val.log'
$errl = Join-Path $wd 'results\purged_val.err.log'

# ---- keep the box awake ---------------------------------------------------
# The previous driver was killed mid-fit when the machine went to sleep, losing
# the run and leaving a stale lock.  ES_CONTINUOUS|ES_SYSTEM_REQUIRED is
# process-scoped: Windows releases it the moment this process exits, so it does
# NOT alter the user's permanent power settings.
try {
    Add-Type -Namespace PvDrv -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction Stop
    [void][PvDrv.Native]::SetThreadExecutionState(0x80000000 -bor 0x00000001)
    Write-Host "keep-awake engaged (process-scoped)"
} catch {
    Write-Host "keep-awake unavailable: $($_.Exception.Message)"
}


function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

$tags = @(
    'leaky:7', 'leaky:2024',
    'purged:42', 'purged:7', 'purged:2024'
)

Log "================ DRIVER START (epochs=$Epochs, retries=$Retries) ================"
$t0 = Get-Date

foreach ($t in $tags) {
    $proto, $seed = $t.Split(':')
    $csv = Join-Path $wd ("results\pv_preds\xLSTM_{0}_s{1}.csv" -f $proto, $seed)
    if (Test-Path $csv) {
        Log "[skip] $t already done"
        continue
    }

    $ok = $false
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        Log "[run ] $t  attempt $attempt/$Retries"
        $sw = [Diagnostics.Stopwatch]::StartNew()
        & $py -u run_purged_val.py --only $t --epochs $Epochs *>> $log 2>> $errl
        $code = $LASTEXITCODE
        $sw.Stop()
        Log ("[done] $t exit=$code in {0:N0}s" -f $sw.Elapsed.TotalSeconds)

        if ((Test-Path $csv) -and $code -eq 0) { $ok = $true; break }

        if ($code -ne 0) {
            Log "[fail] $t exit=$code -- sleeping ${RetrySleepSec}s before retry"
            Start-Sleep -Seconds $RetrySleepSec
        }
    }
    if (-not $ok) { Log "[FAIL] $t gave up after $Retries attempts" }
}

Log "================ COMBINE ================"
& $py -u run_purged_val.py --combine-only *>> $log 2>> $errl
Log "[combine] exit=$LASTEXITCODE"

$dt = (Get-Date) - $t0
Log ("================ DRIVER END in {0:N1} min ================" -f $dt.TotalMinutes)
Remove-Item $lock -ErrorAction SilentlyContinue
