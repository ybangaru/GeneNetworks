import mlflow

mlflow.set_tracking_uri("/data/qd452774/spatial_transcriptomics/mlruns/")
mlflow_client = mlflow.tracking.MlflowClient()