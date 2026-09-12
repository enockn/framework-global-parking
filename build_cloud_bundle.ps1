param(
    [string]$Events = "merged_parking_data.csv",
    [string]$Spaces = "mldf_clean_merge.csv",
    [string]$FrameworkResults = "outputs_pd_evaluation_v9",
    [string]$OutDir = "framework_serving",
    [int]$SamplesPerExt = 360
)
$ErrorActionPreference = "Stop"
Write-Host "Building Framework Streamlit serving bundle..." -ForegroundColor Cyan
python prepare_framework_platform.py `
  --events $Events `
  --spaces $Spaces `
  --framework-results $FrameworkResults `
  --outdir $OutDir `
  --samples-per-ext $SamplesPerExt
if ($LASTEXITCODE -ne 0) { throw "Serving bundle build failed." }
python check_deployment_bundle.py
if ($LASTEXITCODE -ne 0) { throw "Deployment validation failed." }
Write-Host "Serving bundle ready for GitHub/Streamlit." -ForegroundColor Green
