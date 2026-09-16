# Runs risk_fusion.score_live once daily via Windows Task Scheduler.
#
# PYTHONUTF8/PYTHONIOENCODING are required, not optional: Herbie (the HRRR
# fetch library) prints a checkmark on every successful download, and that
# print raises UnicodeEncodeError under the ANSI codepage a non-interactive
# Task Scheduler session falls back to for redirected output - a real
# incident already hit once this way (see backfill_hrrr.py history). Without
# this, every scheduled run would silently fail to fetch anything.
#
# Redirection goes through Start-Process (an OS-level pipe), not PowerShell's
# >>/*>> operators - those re-decode a native process's output through
# PowerShell's own text pipeline, which mangled this exact log into
# double-spaced garbage the first time this script was written.

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$RepoRoot = "M:\_Development\ShowMeFire\model-training"
$LogDir = "M:\_Development\ShowMeFire\training-data\logs\score_live"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Today = Get-Date -Format "yyyy-MM-dd"
$LogFile = Join-Path $LogDir "$Today.log"
$ErrFile = "$LogFile.err"

Set-Location $RepoRoot
$Proc = Start-Process -FilePath "$RepoRoot\.venv\Scripts\python.exe" -ArgumentList "-m", "risk_fusion.score_live" `
    -RedirectStandardOutput $LogFile -RedirectStandardError $ErrFile -NoNewWindow -Wait -PassThru

if ((Get-Item $ErrFile).Length -gt 0) {
    Add-Content -Path $LogFile -Value "`n--- stderr ---" -Encoding UTF8
    Get-Content -Path $ErrFile -Raw | Add-Content -Path $LogFile -Encoding UTF8
}
Remove-Item $ErrFile -ErrorAction SilentlyContinue

Add-Content -Path $LogFile -Value "`n=== exit code: $($Proc.ExitCode) ===" -Encoding UTF8
