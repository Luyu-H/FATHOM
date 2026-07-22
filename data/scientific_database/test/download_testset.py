from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Luyu-H/CMIP6_scientific_database_for_FATHOM",
    repo_type="dataset",
    allow_patterns="test",
    local_dir="data/scientific_database/test"
)