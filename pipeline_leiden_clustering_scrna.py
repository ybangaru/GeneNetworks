import os
import scanpy as sc
from joblib import Parallel, delayed
import mlflow

from graphxl import scRNAPreProcessor, scRNAPipeline, MLFLOW_TRACKING_URI

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def run_parallel(filters_list):
    Parallel(n_jobs=2)(delayed(run_pipeline_clustering)(f) for f in filters_list)


def run_pipeline_clustering(filters):
    data_filter_name = filters["data_filter_name"]
    experiment_name = f"scRNA_clustering_{data_filter_name}"
    experiment_data_filters = filters["data_filters"]
    experiment_data_nan_filter = filters["filter_nans"]
    experiment_state = filters["state"]

    preprocessor_options = {
        "filter_cells": dict(min_genes=200),
        "filter_genes": dict(min_cells=3),
        "find_mt": True,
        "find_rp": True,
        "qc_metrics": dict(qc_vars=["mt", "rp"], percent_top=None, log1p=False, inplace=True),
        "normalize_totals": dict(target_sum=5_000),
        "log1p_transformation": dict(),
        "highly_variable_genes": dict(min_mean=0.01, max_mean=4, min_disp=0.01),
        "scale": dict(zero_center=False),  # , max_value=10
    }

    pca_options = dict(random_state=0, n_comps=240)
    ng_options = dict(n_neighbors=50, random_state=0)
    clustering_options = {
        "umap": dict(n_components=2),
        "louvain": dict(flavor="igraph"),
        "leiden": dict(resolution=1.0, key_added="leiden_res"),
        "paga": dict(),
    }

    if experiment_state == "disease-state":
        data_options = {
            "name": "data/nafld_adata_raw.h5ad",
            "filters": experiment_data_filters,
            "annotations": "/data/yash/liver_cell_atlas/annot_mouseNafldAll.csv",
            "drop_na": experiment_data_nan_filter,
        }
    else:
        data_options = {
            "name": "data/stst_adata_raw.h5ad",
            "filters": experiment_data_filters,
            "annotations": "/data/yash/liver_cell_atlas/annot_mouseStStAll.csv",
            "drop_na": experiment_data_nan_filter,
        }

    mlflow.set_experiment(experiment_name)
    run_name_temp = f"{experiment_state}-using-240-PCs-50-neighbors"
    with mlflow.start_run(run_name=run_name_temp) as run:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

        # set experiment_id and run_id as tags
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("run_name", run_name_temp)

        # add the filters as tags to the run
        mlflow.set_tags({"data_filter": f"{data_filter_name}"})
        mlflow.set_tags({"state": f"{experiment_state}"})

        # log data options
        mlflow.log_params(data_options)

        # log preprocessor options
        mlflow.log_params(preprocessor_options)

        # log PCA options
        mlflow.log_params(pca_options)

        # log neighborhood graph options
        mlflow.log_params(ng_options)

        # log clustering options
        mlflow.log_params(clustering_options)

        pipeline_instance = scRNAPipeline(options=data_options)
        preprocessor_instance = scRNAPreProcessor(options=preprocessor_options)
        pipeline_instance.preprocess_data(preprocessor_instance)
        pipeline_instance.perform_pca(pca_options)
        pipeline_instance.compute_neighborhood_graph(ng_options)
        pipeline_instance.perform_clustering(clustering_options)

        directory_run = f"/home/qd452774/spatial_transcriptomics/data/{experiment_id}/{run_id}"
        if not os.path.exists(directory_run):
            os.makedirs(directory_run)

        if experiment_state == "disease-state":
            h5ad_filename = f"{directory_run}/scRNA_clustering_ds_{data_filter_name}.h5ad"
        else:
            h5ad_filename = f"{directory_run}/scRNA_clustering_ss_{data_filter_name}.h5ad"

        pipeline_instance.data.write_h5ad(h5ad_filename)

        # log the resulting data to mlflow
        mlflow.log_artifact(f"{h5ad_filename}", "data")

        mlflow.end_run()


# TODO: update config to dictionaries from tuples
# experimenent_filters_list = [
#     ("all", "all"),
#     ("diet", "SD"),
#     ("diet", "WD"),
#     ("digest", "exVivo"),
#     ("digest", "inVivo"),
#     ("digest", "nuclei"),
#     ("typeSample", "citeSeq"),
#     ("typeSample", "nucSeq"),
#     ("typeSample", "scRnaSeq"),
# ]

test_case_ds = {
    "data_filter_name": "digest=exVivo&inVivo",
    "data_filters": {"digest": lambda x: x in ["exVivo", "inVivo"]},
    "filter_nans": True,
    "state": "disease-state",
}
test_case_ss = {
    "data_filter_name": "digest=exVivo&inVivo",
    "data_filters": {"digest": lambda x: x in ["exVivo", "inVivo"]},
    "filter_nans": True,
    "state": "steady-state",
}

experimenent_filters_list = [test_case_ds, test_case_ss]
run_parallel(experimenent_filters_list)
