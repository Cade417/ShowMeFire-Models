# Launch the local ShowMeFire Model Lab (Streamlit).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not $env:SMF_DATA_ROOT) {
    $env:SMF_DATA_ROOT = "M:\_Development\ShowMeFire\training-data"
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    & $VenvPython -m streamlit run (Join-Path $PSScriptRoot "app.py")
} else {
    Write-Host "No .venv found under $Root — using current python."
    python -m streamlit run (Join-Path $PSScriptRoot "app.py")
}
