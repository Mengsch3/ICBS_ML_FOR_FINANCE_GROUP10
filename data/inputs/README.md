# Input Data

This folder contains the read-only coursework parquet inputs used by the
pipeline.  The files in this delivery folder were linked from the final working
project data directory on the local machine.

Required files are declared in `config/default.yaml` under `inputs.required_files`.
The pipeline never writes to this folder.  Intermediate transformed data is
created inside `data/outputs/<run_id>/artifacts/Output`.

