param(
    [Parameter(Mandatory=$true)][string]$GitHubRepoUrl,
    [string]$Branch = "main"
)
$ErrorActionPreference = "Stop"
python check_deployment_bundle.py
if ($LASTEXITCODE -ne 0) { throw "Fix the serving bundle before publishing." }

$large = Get-ChildItem "framework_serving" -File -ErrorAction SilentlyContinue | Where-Object { $_.Length -ge 95MB }
if ($large) {
    Write-Host "Large serving artifacts detected; enabling Git LFS..." -ForegroundColor Yellow
    & .\enable_git_lfs.ps1
}

if (-not (Test-Path ".git")) { git init }
git branch -M $Branch
try { git remote remove origin 2>$null } catch {}
git remote add origin $GitHubRepoUrl
git add .
git commit -m "Deploy Framework Global Parking Intelligence Platform" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "No new commit created, or Git identity needs configuration." -ForegroundColor Yellow }
git push -u origin $Branch
if ($LASTEXITCODE -ne 0) { throw "Git push failed. Check GitHub authentication and the repository URL." }
Write-Host "GitHub push complete." -ForegroundColor Green
