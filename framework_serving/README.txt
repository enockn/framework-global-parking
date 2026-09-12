This folder is populated by prepare_framework_platform.py.

Before pushing the repository, run from the repository root:

python prepare_framework_platform.py ^
  --events "merged_parking_data.csv" ^
  --spaces "mldf_clean_merge.csv" ^
  --framework-results "outputs_pd_evaluation_v9" ^
  --outdir "framework_serving"

Do not delete serving_metadata.json after the build. The public Streamlit app
loads only this deployment bundle; the raw multi-million-row research files are
not required on Streamlit Community Cloud.
