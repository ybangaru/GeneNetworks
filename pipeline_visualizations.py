#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import (
    Union,
    Optional,
)
import mlflow
import scanpy as sc
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import plotly.colors.qualitative as pcq
import plotly.colors as pc


pio.templates.default = "plotly_dark"
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=120, facecolor="white")


# In[2]:




class VisualizePipeline:
    def __init__(self, info_tuple, type_fig):
        # super().__init__()
        self.info_cluster = info_tuple
        self.file_name = info_tuple[-1]
        self.type_fig = type_fig
        self.filters = (self.info_cluster[0], self.info_cluster[1])

        self.all_generated_figs = {}
        self.all_generated_umaps = {}
        self.data = self.read_data()
        self.categorical_columns = self.data.obs.select_dtypes(
            include=["category", "object", "bool"]
        ).columns
        self.numerical_columns = self.data.obs.select_dtypes(
            include=["float32", "int32", "float64", "int64"]
        ).columns

    def read_data(self):
        # if not self.data:
        data = sc.read_h5ad(self.file_name)
        return data

    def generate_pca_plots(self):
        for cate_item in self.categorical_columns:
            fig = plotly_pca_categorical(
                self.data,
                self.filters,
                color_key=cate_item,
                return_fig=True,
                x_dim=0,
                y_dim=1,
                show=False,
            )
            self.all_generated_figs[f"{self.type_fig}_{cate_item}"] = fig
        for num_item in self.numerical_columns:
            fig = plotly_pca_numerical(
                self.data,
                self.filters,
                color_key=num_item,
                return_fig=True,
                x_dim=0,
                y_dim=1,
                show=False,
            )
            self.all_generated_figs[f"{self.type_fig}_{num_item}"] = fig

    def generate_umap_plots(self):

        for cate_item in self.categorical_columns:
            fig_true = plotly_umap_categorical(
                self.data,
                self.filters[0],
                self.filters[1],
                color_key=cate_item,
                from_annotations=True,
            )
            self.all_generated_umaps[f"{self.type_fig}_{cate_item}_from_annot"] = fig_true
            fig_false = plotly_umap_categorical(
                self.data,
                self.filters[0],
                self.filters[1],
                color_key=cate_item,
                from_annotations=False,
            )            
            self.all_generated_umaps[f"{self.type_fig}_{cate_item}"] = fig_false
        for num_item in self.numerical_columns:
            fig_true = plotly_umap_numerical(
                self.data,
                self.filters[0],
                self.filters[1],
                color_key=num_item,
                from_annotations=True,
            )
            self.all_generated_umaps[f"{self.type_fig}_{num_item}_from_annot"] = fig_true
            fig_false = plotly_umap_numerical(
                self.data,
                self.filters[0],
                self.filters[1],
                color_key=num_item,
                from_annotations=False,
            )
            self.all_generated_umaps[f"{self.type_fig}_{num_item}"] = fig_false

def plotly_pca_categorical(
    adata,
    filters,
    color_key: str,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    x_dim: int = 0,
    y_dim: int = 1,
):

    if "pca" not in adata.obsm.keys() and "X_pca" not in adata.obsm.keys():
        raise KeyError(
            f"Could not find entry in `obsm` for 'pca'.\n"
            f"Available keys are: {list(adata.obsm.keys())}."
        )

    # Get the PCA coordinates and variance explained
    pca_coords = adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    var_exp = (
        adata.uns["pca"]["variance_ratio"]
        if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys()
        else None
    )

    # Create a dataframe of the PCA coordinates with sample names as the index
    pca_df = pd.DataFrame(
        data=pca_coords[:, [x_dim, y_dim]],
        columns=["PC{}".format(x_dim + 1), "PC{}".format(y_dim + 1)],
        index=adata.obs_names,
    )

    # return pca_df
    # Create a list of colors for each data point
    # nan_mask = np.isnan(pca_df).any(axis=1)
    color_list = adata.obs[color_key].astype(str).replace("nan", "Unknown")
    pca_df["category"] = color_list

    fig = px.scatter(
        x=pca_df.iloc[:, 0],
        y=pca_df.iloc[:, 1],
        color=color_list,
    )

    # Set the axis labels
    x_label = (
        "PC{} ({}%)".format(x_dim + 1, round(var_exp[x_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(x_dim + 1)
    )
    y_label = (
        "PC{} ({}%)".format(y_dim + 1, round(var_exp[y_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(y_dim + 1)
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
        # width = 1200,
        legend={
            "title": f"{color_key} categories",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        title=f"scRNA disease state PCA({color_key} representation) using {filters[0]}={filters[1]} observations",
    )

    fig.update_traces(marker_size=2.5)

    # Show or save the plot
    if save is not None:
        fig.write_html(save)
    if show is not False:
        fig.show()

    # Return the plotly figure object if requested
    if return_fig is True:
        return fig


def plotly_pca_numerical(
    adata,
    filters,
    color_key: str,
    show: Optional[bool] = None,
    return_fig: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    x_dim: int = 0,
    y_dim: int = 1,
):

    if "pca" not in adata.obsm.keys() and "X_pca" not in adata.obsm.keys():
        raise KeyError(
            f"Could not find entry in `obsm` for 'pca'.\n"
            f"Available keys are: {list(adata.obsm.keys())}."
        )

    # Get the PCA coordinates and variance explained
    pca_coords = adata.obsm["pca"] if "pca" in adata.obsm.keys() else adata.obsm["X_pca"]
    var_exp = (
        adata.uns["pca"]["variance_ratio"]
        if "pca" in adata.obsm.keys() or "X_pca" in adata.obsm.keys()
        else None
    )

    # Create a dataframe of the PCA coordinates with sample names as the index
    pca_df = pd.DataFrame(
        data=pca_coords[:, [x_dim, y_dim]],
        columns=["PC{}".format(x_dim + 1), "PC{}".format(y_dim + 1)],
        index=adata.obs_names,
    )

    # Create a list of colors for each data point
    color_list = adata.obs[color_key]

    # Calculate centroids for each category
    # centroids = {}
    # cat_list = color_list.unique()
    # for cat in cat_list:
    #     mask = pca_df[color_key] == cat
    #     centroids[cat] = pca_df[mask].drop(columns=f'{color_key}').mean(axis=0)

    # print(centroids)

    fig = px.scatter(
        x=pca_df.iloc[:, 0],
        y=pca_df.iloc[:, 1],
        color=color_list,
    )

    # Set the axis labels
    x_label = (
        "PC{} ({}%)".format(x_dim + 1, round(var_exp[x_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(x_dim + 1)
    )
    y_label = (
        "PC{} ({}%)".format(y_dim + 1, round(var_exp[y_dim] * 100, 2))
        if var_exp is not None
        else "PC{}".format(y_dim + 1)
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,
        # width = 1200,
        legend={
            "title": f"{color_key} categories",
            "itemsizing": "constant",
            "itemwidth": 30,
        },
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        title=f"scRNA disease state PCA({color_key} representation) using {filters[0]}={filters[1]} observations",
    )

    fig.update_traces(marker_size=2.5)

    # Add annotations to the plot using the category coordinates
    # for category, coords in centroids.items():
    #     fig.add_annotation(
    #         x=coords['PC{}'.format(x_dim+1)],
    #         y=coords['PC{}'.format(y_dim+1)],
    #         text=f'<b><i>{category}</b></i>',
    #         showarrow=False,
    #     )

    # Show or save the plot
    if save is not None:
        fig.write_html(save)
    if show is not False:
        fig.show()

    # Return the plotly figure object if requested
    if return_fig is True:
        return fig


def plotly_umap_categorical(adata, chosen_key, chosen_value, color_key, from_annotations):
    fig=None

    if from_annotations == True:
        umap_coords = adata.obs[['UMAP_1', 'UMAP_2']].values
        color_list = adata.obs[color_key]
        if chosen_key=='all':
            plot_title = f'UMAPs source data, known observations'        
        else:
            plot_title = f'UMAP source data, {chosen_key} - {chosen_value}'

    else:
        umap_coords = adata.obsm['X_umap']
        color_list = adata.obs[color_key].astype(str).replace('nan', 'Unknown')
        if chosen_key=='all':
            plot_title = f'UMAPs constructed, {chosen_key} observations'        
        else:
            plot_title = f'UMAP constructed, {chosen_key} - {chosen_value}'
    
    # Get unique values of color_key
    unique_colors = color_list.unique()
    
    # Get a list of discrete colors from Plotly
    # colors_from_plotly = pcq.D3
    colors_from_plotly = pc.DEFAULT_PLOTLY_COLORS

    # Create a dictionary that maps each unique string to a color
    color_dict = {}
    for i, string in enumerate(unique_colors):
        color_dict[string] = colors_from_plotly[i % len(colors_from_plotly)]

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[
            [{"secondary_y": False}],
        ],
    )

    # Iterate over unique colors to create trace for each color
    for color in unique_colors:
        color_mask = color_list == color
        fig.add_trace(
            go.Scatter(
                x=umap_coords[color_mask, 0],
                y=umap_coords[color_mask, 1],
                mode="markers",
                marker_color=[color_dict[c] for c in color_list[color_mask]],
                name=str(color)
            )
        )

    x_label = 'UMAP1'
    y_label = 'UMAP2'
    fig.update_layout(
        title=f'{plot_title}', height=800, #width=800, 
        xaxis_title=x_label, 
        yaxis_title=y_label, 
        legend= {"title" : f"{color_key} categories", "itemsizing": "constant", "itemwidth": 30},
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )
    fig.update_traces(marker=dict(size=2.5))
    return fig


def plotly_umap_numerical(adata, chosen_key, chosen_value, color_key, from_annotations):
    fig=None

    if from_annotations == True:
        umap_coords = adata.obs[['UMAP_1', 'UMAP_2']].values
        if chosen_key=='all':
            plot_title = f'UMAPs source data, known observations'        
        else:
            plot_title = f'UMAP source data, {chosen_key} - {chosen_value}'

    else:
        umap_coords = adata.obsm['X_umap']
        if chosen_key=='all':
            plot_title = f'UMAPs constructed, {chosen_key} observations'        
        else:
            plot_title = f'UMAP constructed, {chosen_key} - {chosen_value}'


    color_list = adata.obs[color_key]
    # color_list = adata.obs[color_key].astype(str).replace('nan', 'Unknown')

    # Get unique values of color_key
    # unique_colors = color_list.unique()
    
    # Get a list of discrete colors from Plotly
    # colors_from_plotly = pcq.D3
    # colors_from_plotly = pc.DEFAULT_PLOTLY_COLORS

    # Create a dictionary that maps each unique string to a color
    # color_dict = {}
    # for i, string in enumerate(unique_colors):
    #     color_dict[string] = colors_from_plotly[i % len(colors_from_plotly)]

    fig = px.scatter(
        x=umap_coords[:, 0],
        y=umap_coords[:, 1],
        color=color_list,
        labels={'color':f'{color_key}'}
    )

    x_label = 'UMAP1'
    y_label = 'UMAP2'
    fig.update_traces(marker=dict(size=2.5))    
    fig.update_layout(
        title=f'{plot_title}', height=800, #width=800, 
        xaxis_title=x_label, 
        yaxis_title=y_label, 
        legend= {"title" : f"{color_key}", "itemsizing": "constant", "itemwidth": 30},
        xaxis_showgrid=False,
        yaxis_showgrid=False,
    )
    return fig


def run_viz_pipeline(filters):

    # mlflow.set_tracking_uri("/home/qd452774/spatial_transcriptomics/mlruns")
    # mlflow.set_experiment(f"scRNA_clustering_{filters[0]}={filters[1]}")
    
    plots_instance = VisualizePipeline(filters, "PCA")
    # plots_instance.file_name
    # plots_instance.filters
    plots_instance.generate_pca_plots()

    directory_run = f'/home/qd452774/spatial_transcriptomics/data/{filters[3]}/{filters[5]}'
    if not os.path.exists(directory_run):
        os.makedirs(directory_run)
        
    with mlflow.start_run(run_id=filters[5], experiment_id=filters[3]) as run:

        for fig_name in plots_instance.all_generated_figs.keys():
            fig = plots_instance.all_generated_figs[fig_name]
            # Save the figure as an HTML file
            fig.write_html(f"{directory_run}/{fig_name}.html", full_html=False, include_plotlyjs='cdn')

            # mlflow.plotting.plot(div, artifact_file=f"{fig_name}.html", plot_format="html")
            # print(fig_name)
#             with open(f"/home/qd452774/spatial_transcriptomics/data/{fig_name}.html", "w") as f:
#                 f.write(div)

            mlflow.log_artifact(f"{directory_run}/{fig_name}.html", f"PCA")

        mlflow.end_run()
        
def run_umap_pipeline(filters):
    try:
    
        plots_instance = VisualizePipeline(filters, "UMAP")
        # plots_instance.file_name
        # plots_instance.filters
        plots_instance.generate_umap_plots()

        directory_run = f'/home/qd452774/spatial_transcriptomics/data/{filters[3]}/{filters[5]}'
        if not os.path.exists(directory_run):
            os.makedirs(directory_run)

        with mlflow.start_run(run_id=filters[5], experiment_id=filters[3]) as run:

            for fig_name in plots_instance.all_generated_umaps.keys():
                fig = plots_instance.all_generated_umaps[fig_name]
                # Save the figure as an HTML file
                fig.write_html(f"{directory_run}/{fig_name}.html", full_html=False, include_plotlyjs='cdn')

                # mlflow.plotting.plot(div, artifact_file=f"{fig_name}.html", plot_format="html")
                # print(fig_name)
    #             with open(f"/home/qd452774/spatial_transcriptomics/data/{fig_name}.html", "w") as f:
    #                 f.write(div)

                mlflow.log_artifact(f"{directory_run}/{fig_name}.html", f"UMAP")

            mlflow.end_run()
    except:
        pass

def run_parallel(filters_list):
    Parallel(n_jobs=2)(delayed(run_viz_pipeline)(f) for f in filters_list)


# In[ ]:





# In[3]:


# Set the tracking URI to use the local filesystem
mlflow.set_tracking_uri("/home/qd452774/spatial_transcriptomics/mlruns/")
client = mlflow.tracking.MlflowClient()


# In[4]:


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

all_experiments_config = []
for tuple_filter in dst_pca_filters_list:
    my_experiment = f"scRNA_clustering_{tuple_filter[0]}={tuple_filter[1]}"
    my_experiment_id = client.get_experiment_by_name(my_experiment).experiment_id

    my_runs = client.search_runs(experiment_ids=[my_experiment_id])

    for each_run in my_runs:
        try:
            artifact_location_base = f"/home/qd452774/spatial_transcriptomics/mlruns/{my_experiment_id}/{each_run.info.run_uuid}/artifacts"
            
            if each_run.data.tags['state'] == 'disease-state':
                my_filename = f"{artifact_location_base}/data/scRNA_clustering_ds_{each_run.data.tags['data_filter']}.h5ad"
            else:
                my_filename = f"{artifact_location_base}/data/scRNA_clustering_ss_{each_run.data.tags['data_filter']}.h5ad"
                
            all_experiments_config.append(
                (
                    tuple_filter[0],
                    tuple_filter[1],
                    my_experiment,
                    my_experiment_id,
                    each_run.data.tags["run_name"],
                    each_run.info.run_uuid,
                    each_run.data.tags["data_filter"],
                    my_filename,
                )
            )
        except:
            pass


# In[5]:


# len(all_experiments_config)


# In[6]:


# all_experiments_config[-1]


# In[7]:


# all_experiments_config


# In[8]:


run_parallel(all_experiments_config)


# In[9]:


# for experiments_config_item in all_experiments_config:
    # run_viz_pipeline(all_experiments_config[-6])
    # run_viz_pipeline(experiments_config_item)


# In[10]:


# all_experiments_config[-1]


# In[11]:


# run_umap_pipeline(all_experiments_config[-1])


# In[35]:





# In[12]:



# sample_pipeline = VisualizePipeline(all_experiments_config[-1], "UMAP")

# categorical_columns = sample_pipeline.categorical_columns
# chosen_key = all_experiments_config[-1][0]
# chosen_value = all_experiments_config[-1][1]


# In[13]:


# sample_pipeline.categorical_columns


# In[14]:


# sample_pipeline.data


# In[15]:


# categorical_columns


# In[16]:



# chosen_legend=categorical_columns[6]

# fig_true = plotly_umap_categorical(sample_pipeline.data, chosen_key, chosen_value, color_key=chosen_legend, from_annotations=True)#, 'digest', 'typeSample'])
# # fig_true.show()
# fig_false = plotly_umap_categorical(sample_pipeline.data, chosen_key, chosen_value, color_key=chosen_legend, from_annotations=False)#, 'digest', 'typeSample'])
# # fig_false.show()


# In[17]:


# fig_true.show()


# In[18]:


# fig_false.show()


# In[19]:








# numerical_columns = sample_pipeline.numerical_columns


# In[20]:


# numerical_columns


# In[ ]:





# In[21]:


# chosen_legend=numerical_columns[5]

# fig_true = plotly_umap_numerical(sample_pipeline.data, chosen_key, chosen_value, color_key=chosen_legend, from_annotations=True)#, 'digest', 'typeSample'])
# # fig_true.show()
# fig_false = plotly_umap_numerical(sample_pipeline.data, chosen_key, chosen_value, color_key=chosen_legend, from_annotations=False)#, 'digest', 'typeSample'])
# # fig_false.show()


# In[22]:


# fig_true.show()


# In[23]:


# fig_false.show()


# In[24]:


# both_figures = make_subplots(rows=2, cols=1)
# # Add traces from first figure to first subplot
# for trace in fig_true.data:
#     both_figures.add_trace(trace, row=1, col=1)

# # Add traces from second figure to second subplot
# for trace in fig_false.data:
#     both_figures.add_trace(trace, row=2, col=1)

# # Show combined figure
# # both_figures.show()
# # both_figures.add_trace(fig_true, row=1, col=1)
# # both_figures.add_trace(fig_false, row=2, col=1)
# # both_figures.update_layout(height=1600)
# both_figures.show()


# In[ ]:




