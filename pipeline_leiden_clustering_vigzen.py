import os
import ast
from joblib import Parallel, delayed
import mlflow
import squidpy as sq
from graphxl import spatialPipeline, spatialPreProcessor, logger, VisualizePipeline, NO_JOBS


def run_parallel_spatial(filters_list):
    Parallel(n_jobs=NO_JOBS)(delayed(run_pipeline_spatial_clustering)(f) for f in filters_list)


def run_pipeline_spatial_clustering(filters):
    logger.info("Running clustering for")
    logger.info(filters)

    data_filter_name = filters["data_filter_name"]
    experiment_name = f"spatial_clustering_{data_filter_name}"
    experiment_state = filters["state"]
    leiden_resolution = filters["leiden_resolution"]

    data_options = {
        "dir_loc": os.path.realpath(os.path.join(os.getcwd())),
        "names_list": filters["names_list"],
    }
    pp_options = {
        "filter_cells": dict(min_genes=25),
        "filter_genes": dict(min_cells=10),
        "find_mt": True,
        "find_rp": True,
        "var_names_make_unique": True,
        "qc_metrics": dict(qc_vars=["mt", "rp"], percent_top=None, log1p=False, inplace=True),
        "normalize_totals": dict(target_sum=5_000),
        "log1p_transformation": dict(),
        "scale": dict(zero_center=False),
    }

    pca_options = dict(random_state=0, n_comps=220)
    batch_correction_options = dict(key="sample", adjusted_basis="X_pca")
    ng_options = dict(n_neighbors=50, random_state=0)

    logger.info(f"{data_filter_name}")
    logger.info(f"runnin experiment for res {leiden_resolution}")
    clustering_options = {
        "umap": dict(n_components=2),
        "louvain": dict(flavor="igraph"),
        "leiden": dict(resolution=leiden_resolution, key_added="leiden_res"),
    }

    mlflow.set_experiment(experiment_name)
    run_name_temp = f"{experiment_state}-using-220-PCs-50-neighbors-leiden-{leiden_resolution}"
    with mlflow.start_run(run_name=run_name_temp) as run:
        experiment_id = run.info.experiment_id
        run_id = run.info.run_id

        logger.info("Clustering for")
        logger.info(experiment_id)
        logger.info(run_id)

        # set experiment_id and run_id as tags
        mlflow.set_tag("experiment_id", experiment_id)
        mlflow.set_tag("run_id", run_id)
        mlflow.set_tag("run_name", run_name_temp)

        # add the filters as tags to the run
        mlflow.set_tags({"data_filter": f"{data_filter_name}"})
        mlflow.set_tags({"state": f"{experiment_state}"})

        # log data options
        mlflow.log_params(data_options)
        pipeline_instance = spatialPipeline(options=data_options)

        # log preprocessor options
        mlflow.log_params(pp_options)
        preprocessor_instance = spatialPreProcessor(options=pp_options)
        pipeline_instance.preprocess_data(preprocessor_instance)

        # log PCA options
        mlflow.log_params(pca_options)
        pipeline_instance.perform_pca(pca_options)

        if len(filters["names_list"]) > 1:
            # log batch correction params
            mlflow.log_params(batch_correction_options)
            pipeline_instance.perform_batch_correction(batch_correction_options)

        # log neighborhood graph options
        mlflow.log_params(ng_options)
        pipeline_instance.compute_neighborhood_graph(ng_options)

        # log clustering options
        mlflow.log_params(clustering_options)
        pipeline_instance.perform_clustering(clustering_options)

        directory_run = f'{data_options["dir_loc"]}/data/{experiment_id}/{run_id}'
        if not os.path.exists(directory_run):
            os.makedirs(directory_run)

        if experiment_state == "disease-state":
            h5ad_filename = f"{directory_run}/spatial_clustering_ds_{data_filter_name}.h5ad"
            # h5ad_filename_raw = (
            #     f"{directory_run}/spatial_clustering_ss_{data_filter_name}_raw.h5ad"
            # )
        else:
            h5ad_filename = f"{directory_run}/spatial_clustering_ss_{data_filter_name}.h5ad"
            # h5ad_filename_raw = (
            #     f"{directory_run}/spatial_clustering_ss_{data_filter_name}_raw.h5ad"
            # )

        pipeline_instance.data.write_h5ad(h5ad_filename)
        # pipeline_instance.data_raw.write_h5ad(h5ad_filename_raw)

        # log the resulting data to mlflow
        mlflow.log_artifact(f"{h5ad_filename}", "data")
        # mlflow.log_artifact(f"{h5ad_filename_raw}", "data_raw")

        mlflow.end_run()


def run_parallel_visualizations(experiment_runs_config):
    Parallel(n_jobs=NO_JOBS)(delayed(build_visualizations_for_run)(f) for f in experiment_runs_config)


def build_visualizations_for_run(filters):
    experiment_id = filters["experiment_id"]
    run_id = filters["run_id"]

    plots_instance = VisualizePipeline(filters)
    # plots_instance.data.obs['leiden_res'] = plots_instance.data.obs['leiden_res'].map(ANNOTATION_DICT)
    plots_instance.generate_pca_plots()
    plots_instance.generate_umap_plots()

    directory_run = f"/data/qd452774/spatial_transcriptomics/data/{experiment_id}/{run_id}"

    os.makedirs(directory_run, exist_ok=True)
    os.makedirs(f"{directory_run}/PCA", exist_ok=True)
    os.makedirs(f"{directory_run}/UMAP", exist_ok=True)

    logger.info("Creating visualizations for")
    logger.info(directory_run)

    with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:  # noqa
        for fig_name in plots_instance.all_generated_pcas.keys():
            fig = plots_instance.all_generated_pcas[fig_name]
            pca_file_path = f"{directory_run}/PCA/{fig_name}.png"
            fig.write_image(pca_file_path, format="png", width=800, height=800, engine="kaleido")
            # pio.write_image(fig, pca_file_path, format="png", width=800, height=800)
            mlflow.log_artifact(pca_file_path, "PCA/images")
            fig.write_html(
                f"{directory_run}/{fig_name}.html",
                full_html=False,
                include_plotlyjs="cdn",
            )
            mlflow.log_artifact(f"{directory_run}/{fig_name}.html", "PCA")

        for fig_name in plots_instance.all_generated_umaps.keys():
            fig = plots_instance.all_generated_umaps[fig_name]
            umap_file_path = f"{directory_run}/UMAP/{fig_name}.png"
            fig.write_image(umap_file_path, format="png", width=800, height=800, engine="kaleido")
            # pio.write_image(fig, umap_file_path, format="png", width=800, height=800)
            mlflow.log_artifact(umap_file_path, "UMAP/images")
            fig.write_html(
                f"{directory_run}/{fig_name}.html",
                full_html=False,
                include_plotlyjs="cdn",
            )
            mlflow.log_artifact(f"{directory_run}/{fig_name}.html", "UMAP")

        # pio.show_config.close_opened_files = True

        for item in plots_instance.categorical_columns:
            all_slides = plots_instance.data.obs["sample"].unique()

            for slide_sample in all_slides:
                fig_name = f"{item}_{slide_sample}"
                fig = sq.pl.spatial_scatter(
                    plots_instance.data[plots_instance.data.obs["sample"] == slide_sample, :],
                    shape=None,
                    color=item,
                    size=0.5,
                    library_id="spatial",
                    figsize=(13, 13),
                    return_ax=True,
                )
                fig.figure.savefig(f"{directory_run}/{fig_name}.png", dpi=300, bbox_inches="tight")
                # plt.close(fig)
                mlflow.log_artifact(f"{directory_run}/{fig_name}.png", "SpatialScatter")

        for item in plots_instance.numerical_columns:
            all_slides = plots_instance.data.obs["sample"].unique()

            for slide_sample in all_slides:
                fig_name = f"{item}_{slide_sample}"
                fig = sq.pl.spatial_scatter(
                    plots_instance.data[plots_instance.data.obs["sample"] == slide_sample, :],
                    shape=None,
                    color=item,
                    size=0.5,
                    library_id="spatial",
                    figsize=(13, 13),
                    return_ax=True,
                )
                fig.figure.savefig(f"{directory_run}/{fig_name}.png", dpi=300, bbox_inches="tight")
                # plt.close(fig)
                mlflow.log_artifact(f"{directory_run}/{fig_name}.png", "SpatialScatter")


filters_list = [
    # {
    #     "data_filter_name" : "Liver1Slice1",
    #     "names_list" : ["Liver1Slice1"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    # {
    #     "data_filter_name" : "Liver1Slice2",
    #     "names_list" : ["Liver1Slice2"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    # {
    #     "data_filter_name" : "Liver2Slice1",
    #     "names_list" : ["Liver2Slice1"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    # {
    #     "data_filter_name" : "Liver2Slice2",
    #     "names_list" : ["Liver2Slice2"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    # {
    #     "data_filter_name" : "Liver1Slice12",
    #     "names_list" : ["Liver1Slice1", "Liver1Slice2"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    # {
    #     "data_filter_name" : "Liver2Slice12",
    #     "names_list" : ["Liver2Slice1", "Liver2Slice2"],
    #     "state": "steady-state",
    #     "filter_nans" : False,
    # },
    {
        "data_filter_name": "Liver12Slice12",
        "names_list": ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"],
        "state": "steady-state",
        "filter_nans": False,
    },
]
# leiden_filters = []

# lden_resolutions = [0.9, 1.0, 1.1]
# for item in filters_list:
#     for res_ in lden_resolutions:
#         new_item = item.copy()
#         new_item["leiden_resolution"] = res_
#         leiden_filters.append(new_item)

# leiden_resolutions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# for res_ in leiden_resolutions:
#     new_filter = {
#         "data_filter_name": "Liver12Slice12",
#         "names_list": ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"],
#         "state": "steady-state",
#         "filter_nans": False,
#     }
#     new_filter["leiden_resolution"] = res_
#     leiden_filters.append(new_filter)

# run_parallel_spatial(leiden_filters)
# filters = {
#     "data_filter_name" : "Liver12Slice12",
#     "names_list": ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"],
#     "state": "steady-state",
#     "filter_nans" : False,
# }

# run_pipeline_spatial_clustering(filters)

# filters_list.append(
#     {
#         "data_filter_name" : "Liver12Slice12",
#         "names_list": ["Liver1Slice1", "Liver1Slice2", "Liver2Slice1", "Liver2Slice2"],
#         "state": "steady-state",
#         "filter_nans" : False,
#     }
# )

mlflow.set_tracking_uri("/data/qd452774/spatial_transcriptomics/mlruns/")
client = mlflow.tracking.MlflowClient()

experiments_config = []

for experiment_config in filters_list:
    my_experiment = f"spatial_clustering_{experiment_config['data_filter_name']}"
    my_experiment_id = client.get_experiment_by_name(my_experiment).experiment_id

    my_runs = client.search_runs(experiment_ids=[my_experiment_id])

    for each_run in my_runs:
        if (
            "leiden-0.7" in each_run.info.run_name
        ):  # or 'leiden-1.0' in each_run.info.run_name or 'leiden-1.1' in each_run.info.run_name:
            try:
                artifact_location_base = f"/data/qd452774/spatial_transcriptomics/mlruns/{my_experiment_id}/{each_run.info.run_uuid}/artifacts"

                if each_run.data.tags["state"] == "disease-state":
                    my_filename = (
                        f"{artifact_location_base}/data/spatial_clustering_ds_{each_run.data.tags['data_filter']}.h5ad"
                    )
                else:
                    my_filename = (
                        f"{artifact_location_base}/data/spatial_clustering_ss_{each_run.data.tags['data_filter']}.h5ad"
                    )

                experiments_config.append(
                    {
                        "data_filter_name": each_run.data.tags["data_filter"],
                        "state": each_run.data.tags["state"],
                        "experiment": my_experiment,
                        "experiment_id": my_experiment_id,
                        "run_name": each_run.data.tags["run_name"],
                        "run_id": each_run.info.run_uuid,
                        "data_file_name": my_filename,
                        "names_list": ast.literal_eval(each_run.data.params["names_list"]),
                    }
                )
            except Exception as e:
                print(e)
                pass

# run_parallel_visualizations(experiments_config)

run_config_info = experiments_config[0]
build_visualizations_for_run(run_config_info)
