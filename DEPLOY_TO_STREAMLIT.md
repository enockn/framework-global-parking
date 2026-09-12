# Deploy the Framework on GitHub + Streamlit Community Cloud

## 1. Build locally

Do this on the machine that already contains the research datasets and the final Framework results.

```powershell
python -m pip install -r requirements.txt
.\build_cloud_bundle.ps1 -Events "merged_parking_data.csv" -Spaces "mldf_clean_merge.csv" -FrameworkResults "outputs_pd_evaluation_v9"
```

The public cloud app requires `framework_serving/`, not the raw research CSV files.

## 2. Check file sizes

```powershell
python check_deployment_bundle.py
```

GitHub blocks ordinary Git files larger than 100 MB. If the validator reports a large file, install Git LFS and run:

```powershell
.\enable_git_lfs.ps1
```

before `git add`. The `push_to_github.ps1` helper also detects large serving artifacts and enables LFS automatically when available.

## 3. Push to GitHub

Create an empty repository on GitHub, then run:

```powershell
.\push_to_github.ps1 -GitHubRepoUrl "https://github.com/YOUR_USERNAME/framework-global-parking.git"
```

Alternatively use ordinary `git init`, `git add`, `git commit`, `git remote add`, and `git push` commands.

## 4. Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Connect/authorize the GitHub repository.
3. Select **Create app**.
4. Choose the repository and `main` branch.
5. Set the entrypoint to `streamlit_app.py`.
6. In Advanced settings, use a supported Python version compatible with the requirements; Python 3.12 is a good default for this repository.
7. Deploy.

The app uses the repository-relative bundle path `framework_serving`, so no secret is required. If you later move the bundle, set a Streamlit secret:

```toml
FRAMEWORK_BUNDLE_DIR = "path/to/framework_serving"
```

## 5. Updating the live site

Commit and push code or bundle changes to GitHub. Streamlit Community Cloud detects the repository update and redeploys the app automatically.

## Repository contents that belong in GitHub

- `streamlit_app.py`
- `framework_inference.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `framework_serving/` generated deployment assets
- supporting README/deployment scripts

Do **not** push the raw 5+ million row research dataset or the complete modelling output directory.
