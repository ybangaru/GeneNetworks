import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import (
    Collection,
    Union,
    Optional,
    Sequence,
    Any,
    Mapping,
    List,
    Tuple,
)
from matplotlib.axes import Axes
from plotly.subplots import make_subplots

from joblib import Parallel, delayed
import mlflow

# sc.settings.verbosity = 3
# # sc.logging.print_header()
# sc.settings.set_figure_params(dpi=120, facecolor="white")



# class scRNAPipeline:
#     def __init__(self, options):
#         # super().__init__()
#         self.file_name = options["name"]
#         self.filter_config = options["filters"]
#         self.data = self.read_data()

#     def read_data(self):
#         # if not self.data:
#         data = sc.read_h5ad(self.file_name)
#         annotations_data = pd.read_csv(
#             "/data/yash/liver_cell_atlas/annot_mouseNafldAll.csv"
#         )
#         data.obs = pd.merge(
#             data.obs,
#             annotations_data.set_index("cell"),
#             left_index=True,
#             right_index=True,
#             how="left",
#         )
#         if self.filter_config:
#             if self.filter_config[0] == 'all':
#                 pass
#             else:
#                 data = data[data.obs[self.filter_config[0]] == self.filter_config[1], :]
#         return data
#         # else:
#         #     return self.data

#     def preprocess_data(self, proprocessor):
#         proprocessor.extract(self.data)

#     def perform_pca(self, pca_options):
#         sc.tl.pca(self.data, **pca_options)

#     def perform_clustering(self, clustering_options):
#         # Preprocess the data
#         sc.tl.umap(self.data, **clustering_options["umap"])

#         # Louvain clustering
#         sc.tl.louvain(self.data, **clustering_options["louvain"])

#         # Leiden clustering
#         sc.tl.leiden(self.data, **clustering_options["leiden"])

#         # PAGA clustering
#         sc.tl.paga(self.data, **clustering_options["paga"])

#     def compute_neighborhood_graph(self, ng_options):
#         sc.pp.neighbors(self.data, **ng_options)  # , n_pcs=50


# class PreProcessor:
#     def __init__(self, options):
#         self.options = options

#     def extract(self, data):
#         # TODO : move this to plots
#         # sc.pl.highest_expr_genes(data, **self.options=={'n_top' : 30})
#         sc.pp.filter_cells(data, **self.options["filter_cells"])
#         sc.pp.filter_genes(data, **self.options["filter_genes"])

#         if self.options["find_mt"] == True:
#             data.var["mt"] = data.var_names.str.startswith("mt-")
#         if self.options["find_rp"] == True:
#             data.var["rp"] = data.var_names.str.contains("^Rp[sl]")
#         sc.pp.calculate_qc_metrics(data, **self.options["qc_metrics"])
#         sc.pp.normalize_total(data, **self.options["normalize_totals"])
#         sc.pp.log1p(data, **self.options["log1p_transformation"])
#         sc.pp.highly_variable_genes(data, **self.options["highly_variable_genes"])
#         sc.pp.scale(data, **self.options["scale"])

        
# def run_parallel(filters_list):
#     with ThreadPoolExecutor() as executor:
#         executor.map(run_pipeline, filters_list)


def run_parallel(filters_list):
    Parallel(n_jobs=2)(delayed(run_pipeline)(f) for f in filters_list)

def run_pipeline(filters):
    
    # mlflow.set_tracking_uri("/home/qd452774/spatial_transcriptomics/mlruns")
    mlflow.set_experiment(f"scRNA_clustering_{filters[0]}={filters[1]}")

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

    pca_options = dict(random_state=0, n_comps=480)
    ng_options = dict(n_neighbors=75, random_state=0)
    clustering_options = {
        "umap": dict(n_components=2),
        "louvain": dict(flavor="igraph"),
        "leiden": dict(resolution=1.0, key_added="leiden_res"),
        "paga": dict(),
    }

    # data_options = {"name": "data/nafld_adata_raw.h5ad", "filters": dst_pca_filters_list[3]}
    data_options = {"name": "data/nafld_adata_raw.h5ad", "filters": filters}
    
    run_name_temp = "disease-state-using-480-PCs-75-neighbors"
    with mlflow.start_run(run_name=run_name_temp) as run:

        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        
        # set experiment_id and run_id as tags
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("run_name", run_name_temp)
        
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
        preprocessor_instance = PreProcessor(options=preprocessor_options)
        pipeline_instance.preprocess_data(preprocessor_instance)
        pipeline_instance.perform_pca(pca_options)
        pipeline_instance.compute_neighborhood_graph(ng_options)
        pipeline_instance.perform_clustering(clustering_options)
        pipeline_instance.data.write_h5ad(f"data/scRNA_clustering_ds_{filters[0]}={filters[1]}.h5ad")
        # add the filters as tags to the run
        mlflow.set_tags({"data_filter" : f"{filters[0]}={filters[1]}"})
        mlflow.set_tags({f"state": f"disease-state"})

        # log the resulting data to mlflow
        mlflow.log_artifact(f"/home/qd452774/spatial_transcriptomics/data/scRNA_clustering_ds_{filters[0]}={filters[1]}.h5ad", f"data")
        
        mlflow.end_run()


# In[2]:




dst_pca_filters_list = [
    ("all", "all"),
    ("diet", "SD"),
    ("diet", "WD"),
    ("digest", "exVivo"),
    ("digest", "inVivo"),
    ("digest", "nuclei"),
    ("typeSample", "citeSeq"),
    ("typeSample", "nucSeq"),
    ("typeSample", "scRnaSeq"),
]


# run the pipeline in parallel for all filters
run_parallel(dst_pca_filters_list)

# start mlflow
# mlflow.set_tracking_uri("/home/qd452774/spatial_transcriptomics/mlruns")
# mlflow.set_experiment("scRNA_clustering_using_separate_groups_for_diet_digest_typeSample")
# experiment = mlflow.get_experiment_by_name("scRNA_clustering_using_separate_groups_for_diet_digest_typeSample")
# mlflow.start_run(experiment_id=experiment.experiment_id)


# end mlflow
# mlflow.end_run()  

