param(
    [switch]$Install,
    [switch]$Rebuild,
    [string]$Events = "merged_parking_data.csv",
    [string]$Spaces = "mldf_clean_merge.csv",
    [string]$FrameworkResults = "outputs_pd_evaluation_v9",
    [string]$Bundle = "framework_serving"
)
$ErrorActionPreference = "Stop"
if ($Install) {
    Write-Host "Installing Framework platform dependencies ..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt
}
if ($Rebuild -or -not (Test-Path (Join-Path $Bundle "serving_metadata.json"))) {
    & .\build_cloud_bundle.ps1 -Events $Events -Spaces $Spaces -FrameworkResults $FrameworkResults -OutDir $Bundle
}
$env:FRAMEWORK_BUNDLE_DIR = $Bundle
Write-Host "Launching Framework Global Parking Intelligence Platform ..." -ForegroundColor Green
python -m streamlit run streamlit_app.py
