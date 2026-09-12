# Framework Parking Intelligence Platform

A deployment-ready Streamlit interface for the duration-aware urban parking occupancy **Framework** developed and validated in the associated research.

## What the public app provides

- Whole-world interactive basemap with real parking polygons.
- Filters by street, road section, `ext_id`, parking geometry, parking type, and wayside.
- Forecast targets by date/time and the validated Framework horizons.
- Polygon occupancy classes:
  - **Green:** occupancy < 5%
  - **Orange:** 5% <= occupancy <= 50%
  - **Red:** occupancy > 50%
- Historical dates: Framework forecast plus reconstructed actual occupancy where observed.
- Future dates: forecast only; no future actual value is fabricated.
- 90% and 95% uncertainty intervals.

## Scientific serving semantics

The research Framework is state-conditioned. For a future target where the true parking state is not yet observable, the platform marginalizes the saved held-out Framework transition behavior over historically analogous states. For exact held-out historical Framework origins, the exact saved Framework prediction is returned.

Calendar variables are derived from the original `arrival_time` field. For example, `1/25/2023 17:24` yields month 1, Wednesday, and hour 17.4.

## One-time local build

The large raw event files are used **locally** to create a compact deployment bundle. They are not pushed to Streamlit.

```powershell
python -m pip install -r requirements.txt

.\build_cloud_bundle.ps1 `
  -Events "merged_parking_data.csv" `
  -Spaces "mldf_clean_merge.csv" `
  -FrameworkResults "outputs_pd_evaluation_v9"
```

Validate:

```powershell
python check_deployment_bundle.py
```

Run locally:

```powershell
python -m streamlit run streamlit_app.py
```

## GitHub

Create an empty GitHub repository, then from this folder:

```powershell
.\push_to_github.ps1 -GitHubRepoUrl "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git"
```

If the validator finds a serving artifact close to GitHub's normal single-file limit, the push helper automatically enables Git LFS when Git LFS is installed. You can also run `./enable_git_lfs.ps1` manually.

## Streamlit Community Cloud

Deploy the GitHub repository using `streamlit_app.py` as the entrypoint. The repository must contain the generated `framework_serving/` directory.
