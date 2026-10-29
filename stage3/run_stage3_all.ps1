# Driver for Stage 3: one python process per fit (fresh CUDA context), resumable.
#   pwsh -NoProfile -File run_stage3_all.ps1
#   pwsh -NoProfile -File run_stage3_all.ps1 -Epochs 100

param(
    [int]$Epochs = 100,
    [int]$Retries = 5,
    [int]$RetrySleepSec = 60,
    [string]$Cells = 'text_scalar,text_emb,visitor_mtr_text,random4,random6,LSTM_none,LSTM_text_emb,none_k6,visitor_mtr_k6',
    [string]$Seeds = '42,7,2024'
)

$ErrorActionPreference = 'Continue'
$wd = $PSScriptRoot
Set-Location $wd

$CellList = @($Cells -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$SeedList = @($Seeds -split ',' | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ })

$lock = Join-Path $wd 'results\stage3.lock'
if (Test-Path $lock) {
    $old = (Get-Content $lock -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) {
        Write-Host "driver already running as PID $old"; exit 3
    }
}
Set-Content -Path $lock -Value $PID -Encoding UTF8

try {
    Add-Type -Namespace S3 -Name Native -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@ -ErrorAction Stop
    [void][S3.Native]::SetThreadExecutionState(0x80000000 -bor 0x00000001)
} catch { }

$py  = 'C:\Users\Pok fun\Homework\ITP4524_AI_GroupProject\stage0\.venv\Scripts\python.exe'
$log = Join-Path $wd 'results\stage3.log'
$err = Join-Path $wd 'results\stage3.err.log'

function Log([string]$m) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Log "======== STAGE3 DRIVER START epochs=$Epochs cells=$($CellList -join ',') seeds=$($SeedList -join ',') ========"
$t0 = Get-Date

foreach ($cell in $CellList) {
    foreach ($seed in $SeedList) {
        $csv = Join-Path $wd ("results\preds\{0}_s{1}.csv" -f $cell, $seed)
        if (Test-Path $csv) { Log "[skip] ${cell}:$seed"; continue }
        $ok = $false
        for ($a = 1; $a -le $Retries; $a++) {
            Log "[run ] ${cell}:$seed attempt $a/$Retries"
            $sw = [Diagnostics.Stopwatch]::StartNew()
            & $py -u run_stage3.py --only "${cell}:$seed" --epochs $Epochs *>> $log 2>> $err
            $code = $LASTEXITCODE
            $sw.Stop()
            Log ("[done] {0}:{1} exit={2} in {3:N0}s" -f $cell, $seed, $code, $sw.Elapsed.TotalSeconds)
            if ((Test-Path $csv) -and $code -eq 0) { $ok = $true; break }
            if ($code -ne 0) { Start-Sleep -Seconds $RetrySleepSec }
        }
        if (-not $ok) { Log "[FAIL] ${cell}:$seed" }
        [System.GC]::Collect()
        Start-Sleep -Seconds 4
    }
}

Log "======== COMBINE ========"
& $py -u run_stage3.py --combine-only *>> $log 2>> $err
Log "[combine] exit=$LASTEXITCODE"
$dt = (Get-Date) - $t0
Log ("======== STAGE3 DRIVER END in {0:N1} min ========" -f $dt.TotalMinutes)
Remove-Item $lock -ErrorAction SilentlyContinue
