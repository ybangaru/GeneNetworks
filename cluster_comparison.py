import os
from joblib import Parallel, delayed
import pandas as pd
import mlflow
import scanpy as sc
from helpers import logger
import plotly.express as px

# Read the SLURM_CPUS_PER_TASK environment variable
SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK")
# Set the number of jobs to the value specified in SLURM_CPUS_PER_TASK
NO_JOBS = int(SLURM_CPUS_PER_TASK) if SLURM_CPUS_PER_TASK is not None else 1


mlflow.set_tracking_uri("/data/qd452774/spatial_transcriptomics/mlruns/")
client = mlflow.tracking.MlflowClient()


def get_run_info(x_experiment, x_resolution):

    xexp_name = f"spatial_clustering_{x_experiment}"
    xrun_name = f"steady-state-using-220-PCs-50-neighbors-leiden-{x_resolution}"

    # Get the experiment ID by name
    experiment = client.get_experiment_by_name(xexp_name)
    experiment_id = experiment.experiment_id

    # Search for the run by run name within the specified experiment
    runs = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.run_name='{xrun_name}'")

    # Check if any runs match the criteria
    if len(runs) > 1:
        logger.error("more runs are there with same name than expected")
        return None
    elif len(runs) == 1:
        # run = runs.iloc[0]  # Assuming there is only one matching run
        # run_id = run.run_id
        return runs[0]
    else:    
        logger.debug(f"No matching run found for run name: {xrun_name} in experiment: {xexp_name}")
        return None


def read_run_result_ann_data(data_filter_name, x_resolution):

    xrun_info = get_run_info(data_filter_name, x_resolution)
    exp_id = xrun_info.info.experiment_id
    run_id = xrun_info.info.run_id
    x_anndata_path = f"/data/qd452774/spatial_transcriptomics/mlruns/{exp_id}/{run_id}/artifacts/data/spatial_clustering_ss_{data_filter_name}.h5ad"
    x_data = sc.read_h5ad(x_anndata_path)
    return x_data


def build_comparison_heatmap(x_data, y_data, x_data_filter_name, y_data_filter_name, x_resolution, y_resolution, transformation=None):

    # Convert the Series to DataFrames
    x_data_df = pd.DataFrame({'x_category': x_data.obs.leiden_res})
    y_data_df = pd.DataFrame({'y_category': y_data.obs.leiden_res})

    # Create an empty DataFrame to store cell IDs
    heatmap_data = pd.DataFrame(index=x_data_df['x_category'].cat.categories, columns=y_data_df['y_category'].cat.categories)

    # Create dictionaries to store common and missing cell IDs
    common_cells_dict = {}
    excess_cells_dict = {}

    # Iterate through the data and fill in the DataFrame and dictionaries
    for x_cat in heatmap_data.index:
        for y_cat in heatmap_data.columns:
            x_cells = x_data_df[x_data_df['x_category'] == x_cat].index
            y_cells = y_data_df[y_data_df['y_category'] == y_cat].index
            common_cells = list(set(x_cells) & set(y_cells))
            excess_cells_x = list(set(x_cells) - set(y_cells))
            excess_cells_y = list(set(y_cells) - set(x_cells))

            # Store common and missing cell IDs in dictionaries
            common_cells_dict[(x_cat, y_cat)] = common_cells
            excess_cells_dict[(x_cat, y_cat)] = (excess_cells_x, excess_cells_y)

            heatmap_data.at[x_cat, y_cat] = len(common_cells)

    # Fill NaN values with zeros
    heatmap_data = heatmap_data.fillna(0)


    # Create the heatmap using Plotly
    x_label = f"{x_data_filter_name} - leiden clusters - {x_resolution} resolution"
    y_label = f"{y_data_filter_name} - leiden clusters - {y_resolution} resolution"

    if transformation == "vertical-percentage":
        given_title = "common cell counts (percentage obtained along the vertical axis)"
        heatmap_data = heatmap_data / heatmap_data.sum(axis=0)
        heatmap_data = heatmap_data * 100
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            labels=dict(x=y_label, y=x_label),
            title=given_title, 
            zmin=0,
            zmax=100,      
        )        
    else:
        given_title = "common cell counts"
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            labels=dict(x=y_label, y=x_label),
            title=given_title, 
        )        

    # Customize the layout
    fig.update_layout(
        xaxis=dict(side="top"),
        width=800,
        height=800,
        # margin=dict(
        #     # t=10,  # Adjust this value to control the margin at the top
        #     # b=50,   # You can also adjust the margin at the bottom if needed
        # ),        
    )

    return fig


# y_data_filter_name = "Liver12Slice12"
# y_resolution = 0.6
# y_data = read_run_result_ann_data(y_data_filter_name, y_resolution)

# x_data_filter_name = "Liver2Slice1"
# x_resolution = 0.6
# x_data = read_run_result_ann_data(x_data_filter_name, x_resolution)

# # print(heatmap_data)

# transformation = "vertical-percentage"
# fig = build_comparison_heatmap(x_data, y_data, x_data_filter_name, y_data_filter_name, x_resolution, y_resolution, transformation)
# fig.show()

# # Now you can access the common_cells_dict and missing_cells_dict to get the cell IDs for each category combination.
# # print(x_data)

def run_parallel_comparison(combinations):
    logger.info(f"{NO_JOBS} jobs are being used")
    Parallel(n_jobs=NO_JOBS)(delayed(build_cluster_comparison)(f) for f in combinations)


def build_cluster_comparison(info_tuple):

    y_data_filter_name = info_tuple[0]
    y_resolution = info_tuple[1]
    x_data_filter_name = info_tuple[2]
    x_resolution = info_tuple[3]

    logger.info(f"{x_data_filter_name} - {x_resolution}")
    logger.info(f"{y_data_filter_name} - {y_resolution}")

    x_data = read_run_result_ann_data(x_data_filter_name, x_resolution)
    y_data = read_run_result_ann_data(y_data_filter_name, y_resolution)

    directory_run = f"/data/qd452774/spatial_transcriptomics/assets/cluster_comparison"

    fig = build_comparison_heatmap(x_data, y_data, x_data_filter_name, y_data_filter_name, x_resolution, y_resolution)
    fig_name = f"{x_data_filter_name}={x_resolution}-{y_data_filter_name}={y_resolution}"
    fig.write_html(
        f"{directory_run}/{fig_name}.html",
        full_html=False,
        include_plotlyjs="cdn",
    )

    transformation = "vertical-percentage"
    fig_transformation = build_comparison_heatmap(x_data, y_data, x_data_filter_name, y_data_filter_name, x_resolution, y_resolution, transformation)
    fig_name_transformation = f"{x_data_filter_name}={x_resolution}-{y_data_filter_name}={y_resolution}_vertical_percent"
    fig_transformation.write_html(
        f"{directory_run}/{fig_name_transformation}.html",
        full_html=False,
        include_plotlyjs="cdn",
    )


def main():
    logger.info("Starting the script")

    x_experiments = ["Liver12Slice12"]
    x_resolutions = [0.7] #[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    y_experiments = ["Liver1Slice1", "Liver1Slice2", "Liver1Slice12", "Liver2Slice1", "Liver2Slice2", "Liver2Slice12"]
    y_resolutions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

    exp_config = []

    for exp in x_experiments:
        for resol in x_resolutions:
            exp_config.append((exp, resol))


    final_config = []
    for exp_x in exp_config:
        for exp_y in y_experiments:
            for resol_y in y_resolutions:
                final_config.append((*exp_x, exp_y, resol_y))

    logger.info(f"Number of combinations - {len(final_config)}")
    # print(final_config)

    build_cluster_comparison(final_config[0])

    # run_parallel_comparison(final_config)
    logger.info("Script completed")
    # data_filter_name = "Liver1Slice2"
    # resolution = 0.6
    # ann_data = read_run_result_ann_data(data_filter_name, resolution)
    # print(ann_data)


if __name__ == "__main__":
    main()