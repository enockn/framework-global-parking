$ErrorActionPreference = "Stop"
Write-Host "Enabling Git LFS for large Framework serving artifacts..." -ForegroundColor Cyan
git lfs install
if ($LASTEXITCODE -ne 0) { throw "Git LFS is not available. Install Git LFS first." }
git lfs track "framework_serving/*.npz"
git lfs track "framework_serving/*.parquet"
git lfs track "framework_serving/*.csv.gz"
git lfs track "framework_serving/*.geojson"
Write-Host "Git LFS tracking rules written to .gitattributes." -ForegroundColor Green
