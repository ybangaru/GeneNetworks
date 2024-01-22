#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:12:57 2021

@author: zqwu
"""
import os
import math
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import networkx as nx
import warnings
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
import geopandas as gpd
from rtree import index
from shapely.geometry import Polygon, LineString

RADIUS_RELAXATION = 0.1
NEIGHBOR_EDGE_CUTOFF = 55  # distance cutoff for neighbor edges, 55 pixels~20 um


def plot_voronoi_polygons(voronoi_polygons, voronoi_polygon_colors=None):
    """Plot voronoi polygons for the cellular graph

    Args:
        voronoi_polygons (nx.Graph/list): cellular graph or list of voronoi polygons
        voronoi_polygon_colors (list): list of colors for voronoi polygons
    """
    if isinstance(voronoi_polygons, nx.Graph):
        voronoi_polygons = [voronoi_polygons.nodes[n]["voronoi_polygon"] for n in voronoi_polygons.nodes]

    if voronoi_polygon_colors is None:
        voronoi_polygon_colors = ["w"] * len(voronoi_polygons)
    assert len(voronoi_polygon_colors) == len(voronoi_polygons)

    xmax = 0
    ymax = 0
    for polygon, polygon_color in zip(voronoi_polygons, voronoi_polygon_colors):
        x, y = polygon[:, 0], polygon[:, 1]
        plt.fill(x, y, facecolor=polygon_color, edgecolor="k", linewidth=0.5)
        xmax = max(xmax, x.max())
        ymax = max(ymax, y.max())

    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    return


def plot_graph(G, node_colors=None):
    """Plot dot-line graph for the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        node_colors (list): list of node colors. Defaults to None.
    """
    # Extract basic node attributes
    node_coords = [G.nodes[n]["center_coord"] for n in G.nodes]
    node_coords = np.stack(node_coords, 0)

    if node_colors is None:
        unique_cell_types = sorted(set([G.nodes[n]["cell_type"] for n in G.nodes]))
        cell_type_to_color = {ct: matplotlib.cm.get_cmap("tab20")(i % 20) for i, ct in enumerate(unique_cell_types)}
        node_colors = [cell_type_to_color[G.nodes[n]["cell_type"]] for n in G.nodes]
    assert len(node_colors) == node_coords.shape[0]

    for i, j, edge_type in G.edges.data():
        xi, yi = G.nodes[i]["center_coord"]
        xj, yj = G.nodes[j]["center_coord"]
        if edge_type["edge_type"] == "neighbor":
            plotting_kwargs = {"c": "k", "linewidth": 1, "linestyle": "-"}
        else:
            plotting_kwargs = {"c": (0.4, 0.4, 0.4, 1.0), "linewidth": 0.3, "linestyle": "--"}
        plt.plot([xi, xj], [yi, yj], zorder=1, **plotting_kwargs)

    plt.scatter(node_coords[:, 0], node_coords[:, 1], s=10, c=node_colors, linewidths=0.3, zorder=2)
    plt.xlim(0, node_coords[:, 0].max() * 1.01)
    plt.ylim(0, node_coords[:, 1].max() * 1.01)
    return


def load_cell_coords(cell_coords_file):
    """Load cell coordinates from file

    Args:
        cell_coords_file (str): path to csv file containing cell coordinates

    Returns:
        pd.DataFrame: dataframe containing cell coordinates, columns ['CELL_ID', 'X', 'Y']
    """
    df = pd.read_csv(cell_coords_file)
    df.columns = [c.upper() for c in df.columns]
    assert "X" in df.columns, "Cannot find column for X coordinates"
    assert "Y" in df.columns, "Cannot find column for Y coordinates"
    if "CELL_ID" not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df["CELL_ID"] = df.index
    return df[["CELL_ID", "X", "Y"]]


def load_cell_types(cell_types_file):
    """Load cell types from file

    Args:
        cell_types_file (str): path to csv file containing cell types

    Returns:
        pd.DataFrame: dataframe containing cell types, columns ['CELL_ID', 'CELL_TYPE']
    """
    df = pd.read_csv(cell_types_file)
    df.columns = [c.upper() for c in df.columns]

    cell_type_column = [c for c in df.columns if c != "CELL_ID"]
    if len(cell_type_column) == 1:
        cell_type_column = cell_type_column[0]
    elif "CELL_TYPE" in cell_type_column:
        cell_type_column = "CELL_TYPE"
    elif "CELL_TYPES" in cell_type_column:
        cell_type_column = "CELL_TYPES"
    else:
        raise ValueError("Please rename the column for cell type as 'CELL_TYPE'")

    if "CELL_ID" not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df["CELL_ID"] = df.index
    _df = df[["CELL_ID", cell_type_column]]
    _df.columns = ["CELL_ID", "CELL_TYPE"]  # rename columns for clarity
    return _df


def load_cell_biomarker_expression(cell_biomarker_expression_file):
    """Load cell biomarker expression from file

    Args:
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression

    Returns:
        pd.DataFrame: dataframe containing cell biomarker expression,
            columns ['CELL_ID', 'BM-<biomarker1_name>', 'BM-<biomarker2_name>', ...]
    """
    df = pd.read_csv(cell_biomarker_expression_file)
    df.columns = [c.upper() for c in df.columns]
    biomarkers = sorted([c for c in df.columns if c != "CELL_ID"])
    for bm in biomarkers:
        if df[bm].dtype not in [np.dtype(int), np.dtype(float), np.dtype("float64")]:
            warnings.warn("Skipping column %s as it is not numeric" % bm)
            biomarkers.remove(bm)

    if "CELL_ID" not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df["CELL_ID"] = df.index
    _df = df[["CELL_ID"] + biomarkers]
    _df.columns = ["CELL_ID"] + ["BM-%s" % bm for bm in biomarkers]
    return _df


def load_cell_features(cell_features_file):
    """Load additional cell features from file

    Args:
        cell_features_file (str): path to csv file containing additional cell features

    Returns:
        pd.DataFrame: dataframe containing cell features
            columns ['CELL_ID', '<feature1_name>', '<feature2_name>', ...]
    """
    df = pd.read_csv(cell_features_file)
    df.columns = [c.upper() for c in df.columns]

    feature_columns = sorted([c for c in df.columns if c != "CELL_ID"])
    for feat in feature_columns:
        if df[feat].dtype not in [np.dtype(int), np.dtype(float), np.dtype("float64")]:
            warnings.warn("Skipping column %s as it is not numeric" % feat)
            feature_columns.remove(feat)

    if "CELL_ID" not in df.columns:
        warnings.warn("Cannot find column for cell id, using index as cell id")
        df["CELL_ID"] = df.index

    return df[["CELL_ID"] + feature_columns]


def read_raw_voronoi(voronoi_file):
    """Read raw coordinates of voronoi polygons from file

    Args:
        voronoi_file (str): path to the voronoi polygon file

    Returns:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
    """
    if voronoi_file.endswith("json"):
        with open(voronoi_file) as f:
            raw_voronoi_polygons = json.load(f)
    elif voronoi_file.endswith(".pkl"):
        with open(voronoi_file, "rb") as f:
            raw_voronoi_polygons = pickle.load(f)

    voronoi_polygons = []
    for i, polygon in enumerate(raw_voronoi_polygons):
        if isinstance(polygon, list):
            polygon = np.array(polygon).reshape((-1, 2))
        elif isinstance(polygon, dict):
            assert len(polygon) == 1
            polygon = list(polygon.values())[0]
            polygon = np.array(polygon).reshape((-1, 2))
        voronoi_polygons.append(polygon)
    return voronoi_polygons


def calcualte_voronoi_from_coords(x, y, xmax=None, ymax=None, xmin=None, ymin=None):
    """Calculate voronoi polygons from a set of points

    Points are assumed to have coordinates in ([0, xmax], [0, ymax])

    Args:
        x (array-like): x coordinates of points
        y (array-like): y coordinates of points
        xmax (float): maximum x coordinate
        ymax (float): maximum y coordinate

    Returns:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
    """
    from geovoronoi import voronoi_regions_from_coords
    from shapely import geometry

    xmax = 1.01 * max(x) if xmax is None else xmax
    ymax = 1.01 * max(y) if ymax is None else ymax
    xmin = 0.99 * min(x) if xmin is None else xmin
    ymin = 0.99 * min(y) if ymin is None else ymin
    boundary = geometry.Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

    coords = np.stack([np.array(x).reshape((-1,)), np.array(y).reshape((-1,))], 1)
    region_polys, region_pts = voronoi_regions_from_coords(coords, boundary)
    # sort region_pts to align with input coords
    sorted_region_pts = {k: v for k, v in sorted(region_pts.items(), key=lambda item: item[1])}
    voronoi_polygons = [np.array(list(region_polys[k].exterior.coords)) for k in sorted_region_pts]

    return voronoi_polygons


def build_graph_from_cell_coords(cell_data, cell_boundaries, boundary_augments, edge_config):
    """Construct a networkx graph based on cell coordinates

    Args:
        cell_data: annData object containing all samples chosen
        cell_boundaries (list): list of boundaries of cells,
            represented by the coordinates of their exterior vertices

    Returns:
        G (nx.Graph): full cellular graph of the region
    """

    edge_logic = edge_config["type"]

    coord_ar = np.array(cell_data[["CELL_ID", "X", "Y"]])
    G = nx.Graph()
    node_to_cell_mapping = {}

    if edge_logic == "Delaunay":
        for i, row in enumerate(coord_ar):
            G.add_node(i, **{boundary_augments: cell_boundaries[i]})
            node_to_cell_mapping[i] = row[0]

        dln = Delaunay(coord_ar[:, 1:3])
        neighbors = [set() for _ in range(len(coord_ar))]
        for t in dln.simplices:
            for v in t:
                neighbors[v].update(t)

        for i, ns in enumerate(neighbors):
            for n in ns:
                G.add_edge(int(i), int(n))

    elif edge_logic == "R3Index":
        # Create a GeoDataFrame with the geometry column
        gdf = gpd.GeoDataFrame(geometry=[Polygon(coords) for coords in cell_boundaries])
        # Merge the GeoDataFrame with cell_data based on index
        gdf = gdf.merge(cell_data, left_index=True, right_index=True)
        # Create an R-tree spatial index
        spatial_index = index.Index()

        # Populate the spatial index with bounding boxes and cell indices
        for idx, geometry in enumerate(gdf.geometry):
            spatial_index.insert(idx, geometry.bounds)

        # for i, row in enumerate(gdf):
        for i, cell in gdf.iterrows():
            # Create a rectangle polygon from the bounds
            if edge_config["bound_type"] == "rectangle":
                temp_bounds = list(cell.geometry.bounds)
                rectangle = [
                    [temp_bounds[0], temp_bounds[1]],
                    [temp_bounds[0], temp_bounds[3]],
                    [temp_bounds[2], temp_bounds[3]],
                    [temp_bounds[2], temp_bounds[1]],
                ]
                transformed_rectangle = np.array(rectangle)
            elif edge_config["bound_type"] == "rotated_rectangle":
                transformed_rectangle = np.array(list(cell.geometry.minimum_rotated_rectangle.exterior.coords))

            G.add_node(
                i, **{boundary_augments: cell_boundaries[i]}, **{edge_config["bound_type"]: transformed_rectangle}
            )
            node_to_cell_mapping[i] = cell["CELL_ID"]

        # Iterate through cells and find nearby candidates using the spatial index
        threshold_distance = edge_config["threshold_distance"]
        for i, cell in gdf.iterrows():
            candidate_indices = list(spatial_index.intersection(cell.geometry.buffer(threshold_distance).bounds))
            for idx in candidate_indices:
                if idx != i:
                    G.add_edge(i, idx)

    elif edge_logic == "MST":
        # Calculate all pairwise distances
        distances = cdist(coord_ar[:, 1:3].astype(float), coord_ar[:, 1:3].astype(float))

        # Create a complete graph
        for i, row in enumerate(coord_ar):
            G.add_node(i, **{boundary_augments: cell_boundaries[i]})
            node_to_cell_mapping[i] = row[0]

        # Add edges with weights
        for i in range(len(coord_ar)):
            for j in range(i + 1, len(coord_ar)):
                G.add_edge(i, j, weight=distances[i, j])

        # Compute the minimum spanning tree
        G = nx.minimum_spanning_tree(G)

    return G, node_to_cell_mapping


def build_graph_from_voronoi_polygons(voronoi_polygons, radius_relaxation=RADIUS_RELAXATION):
    """Construct a networkx graph based on voronoi polygons

    Args:
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    G = nx.Graph()

    polygon_vertices = []
    vertice_identities = []
    for i, polygon in enumerate(voronoi_polygons):
        G.add_node(i, voronoi_polygon=polygon)
        polygon_vertices.append(polygon)
        vertice_identities.append(np.ones((polygon.shape[0],)) * i)

    polygon_vertices = np.concatenate(polygon_vertices, 0)
    vertice_identities = np.concatenate(vertice_identities, 0).astype(int)
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        points_inside = np.where(
            path.contains_points(polygon_vertices, radius=radius_relaxation)
            + path.contains_points(polygon_vertices, radius=-radius_relaxation)
        )[0]
        id_inside = set(vertice_identities[points_inside])
        for j in id_inside:
            if j > i:
                G.add_edge(int(i), int(j))
    return G


def build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data):
    """Construct 1-to-1 mapping between voronoi polygons and cells

    Args:
        G (nx.Graph): full cellular graph of the region
        voronoi_polygons (list): list of voronoi polygons,
            represented by the coordinates of their exterior vertices
        cell_data (pd.DataFrame): dataframe containing cellular data

    Returns:
        voronoi_polygon_to_cell_mapping (dict): 1-to-1 mapping between
            polygon index (also node index in `G`) and cell id
    """
    cell_coords = np.array(list(zip(cell_data["X"], cell_data["Y"]))).reshape((-1, 2))
    # Fetch all cells within each polygon
    cells_in_polygon = {}
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        _cell_ids = cell_data.iloc[np.where(path.contains_points(cell_coords))[0]]
        _cells = list(_cell_ids[["CELL_ID", "X", "Y"]].values)
        cells_in_polygon[i] = _cells

    def get_point_reflection(c1, c2, c3):
        # Reflection of point c1 across line defined by c2 & c3
        x1, y1 = c1
        x2, y2 = c2
        x3, y3 = c3
        if x2 == x3:
            return (2 * x2 - x1, y1)
        m = (y3 - y2) / (x3 - x2)
        c = (x3 * y2 - x2 * y3) / (x3 - x2)
        d = (float(x1) + (float(y1) - c) * m) / (1 + m**2)
        x4 = 2 * d - x1
        y4 = 2 * d * m - y1 + 2 * c
        return (x4, y4)

    # Establish 1-to-1 mapping between polygons and cell ids
    voronoi_polygon_to_cell_mapping = {}
    for i, polygon in enumerate(voronoi_polygons):
        path = mplPath.Path(polygon)
        if len(cells_in_polygon[i]) == 1:
            # A single polygon contains a single cell centroid, assign cell id
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][0][0]

        elif len(cells_in_polygon[i]) == 0:
            # Skipping polygons that do not contain any cell centroids
            continue

        else:
            # A single polygon contains multiple cell centroids
            polygon_edges = [(polygon[_i], polygon[_i + 1]) for _i in range(-1, len(polygon) - 1)]
            # Use the reflection of neighbor polygon's center cell
            neighbor_cells = sum([cells_in_polygon[j] for j in G.neighbors(i)], [])
            reflection_points = np.concatenate(
                [
                    [get_point_reflection(cell[1:], edge[0], edge[1]) for edge in polygon_edges]
                    for cell in neighbor_cells
                ],
                0,
            )
            reflection_points = reflection_points[np.where(path.contains_points(reflection_points))]
            # Reflection should be very close to the center cell
            dists = [((reflection_points - c[1:]) ** 2).sum(1).min(0) for c in cells_in_polygon[i]]
            if not np.min(dists) < 0.01:
                warnings.warn("Cannot find the exact center cell for polygon %d" % i)
            voronoi_polygon_to_cell_mapping[i] = cells_in_polygon[i][np.argmin(dists)][0]
    return voronoi_polygon_to_cell_mapping


def calculate_sbr_orientation(boundary_polygon):
    # Get the minimum rotated rectangle (SBR)
    sbr = boundary_polygon.minimum_rotated_rectangle

    # Get coordinates of SBR vertices
    x, y = sbr.exterior.coords.xy

    # Calculate the differences between x and y coordinates of two consecutive vertices
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Calculate the orientation (angle) of the SBR
    sbr_orientation = math.degrees(math.atan2(dy, dx)) % 180

    return sbr_orientation


def calculate_most_frequent_wall_orientation(boundary_polygon, tolerance):
    # Get the exterior of the polygon
    exterior = boundary_polygon.exterior

    # Get the coordinates of the exterior
    coords = list(exterior.coords)

    # Calculate the orientations of the walls
    wall_orientations = []
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        angle = math.degrees(math.atan2(dy, dx)) % 180
        wall_orientations.append(angle)

    # Create a histogram with a bin size based on the tolerance
    bins = np.arange(0, 180, tolerance)
    hist, bin_edges = np.histogram(wall_orientations, bins=bins)

    # Find the bin with the maximum count
    max_bin_index = np.argmax(hist)

    # Calculate the orientation corresponding to the maximum bin
    most_frequent_orientation = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    return most_frequent_orientation


def assign_attributes(
    G, cell_data, cell_boundaries, node_to_cell_mapping, neighbor_edge_cutoff=NEIGHBOR_EDGE_CUTOFF, padding_dict=None
):
    """Assign node and edge attributes to the cellular graph

    Args:
        G (nx.Graph): full cellular graph of the region
        cell_data (pd.DataFrame): dataframe containing cellular data
        node_to_cell_mapping (dict): 1-to-1 mapping between
            node index in `G` and cell id

    Returns:
        nx.Graph: populated cellular graph
    """
    assert set(G.nodes) == set(node_to_cell_mapping.keys())
    biomarkers = sorted([c for c in cell_data.columns if c.startswith("BM-")])

    additional_features = sorted(
        [c for c in cell_data.columns if c not in biomarkers + ["CELL_ID", "X", "Y", "CELL_TYPE"]]
    )

    cell_to_node_mapping = {v: k for k, v in node_to_cell_mapping.items()}
    node_properties = {}
    for _, cell_row in cell_data.iterrows():
        cell_id = cell_row["CELL_ID"]
        if cell_id not in cell_to_node_mapping:
            continue
        node_index = cell_to_node_mapping[cell_id]
        p = {"cell_id": cell_id}
        p["center_coord"] = (cell_row["X"], cell_row["Y"])

        if padding_dict:
            p["is_padding"] = padding_dict[cell_id]
        if "CELL_TYPE" in cell_row:
            p["cell_type"] = cell_row["CELL_TYPE"]
        else:
            p["cell_type"] = "Unassigned"
        biomarker_expression_dict = {bm.split("BM-")[1]: cell_row[bm] for bm in biomarkers}
        p["biomarker_expression"] = biomarker_expression_dict
        for feat_name in additional_features:
            p[feat_name] = cell_row[feat_name]
        p["boundary_polygon"] = cell_boundaries[p["cell_id"]][0]
        boundary_polygon = Polygon(p["boundary_polygon"])
        # Calculating Size (using the bounding box diagonal)
        # p["poly_diag_size"] = np.linalg.norm(boundary_polygon.bounds[2:] - boundary_polygon.bounds[:2])
        # Calculating Perimeter
        p["perimeter"] = boundary_polygon.length
        # Calculating Area
        p["area"] = boundary_polygon.area
        # Calculating Mean radius
        # TODO: understand centroid variation given vs generated
        # centroid = boundary_polygon.centroid
        p["mean_radius"] = np.mean(
            [Point(p["center_coord"]).distance(Point(vertex)) for vertex in boundary_polygon.exterior.coords]
        )
        # Calculate the Compactness/Circularity
        p["compactness_circularity"] = 4 * np.pi * boundary_polygon.area / boundary_polygon.length**2
        # Calculate the Fractality
        p["fractality"] = np.log(boundary_polygon.area) / np.log(boundary_polygon.length)
        # Calculate the Concavity (Area ratio of the building to its convex hull)
        p["concavity"] = boundary_polygon.area / boundary_polygon.convex_hull.area
        # Calculate the Elongation (Length-width ratio of the building SBR)
        width = (
            boundary_polygon.minimum_rotated_rectangle.bounds[2] - boundary_polygon.minimum_rotated_rectangle.bounds[0]
        )
        p["elongation"] = boundary_polygon.minimum_rotated_rectangle.length / width
        # Get the area of the building
        building_area = boundary_polygon.area
        # Calculate the radius of an equal-area circle
        equal_area_circle_radius = np.sqrt(building_area / np.pi)
        # Create a circle with the same area as the building
        equal_area_circle = Point(p["center_coord"]).buffer(equal_area_circle_radius)
        # Calculate the Overlap Index (Area ratio of the intersection and union between the building and its equal area circle)
        p["overlap_index"] = (
            boundary_polygon.intersection(equal_area_circle).area / boundary_polygon.union(equal_area_circle).area
        )
        # SBR Orientation
        p["sbro_orientation"] = calculate_sbr_orientation(boundary_polygon)
        # most frequent wall orientation
        p["wswo_orientation"] = calculate_most_frequent_wall_orientation(boundary_polygon, 10)

        node_properties[node_index] = p

    nx.set_node_attributes(G, node_properties)

    # Add distance, edge type (by thresholding) to edge feature
    edge_properties = get_edge_type(G, neighbor_edge_cutoff=neighbor_edge_cutoff)
    nx.set_edge_attributes(G, edge_properties)
    return G


def get_edge_type(G, neighbor_edge_cutoff):
    """Define neighbor vs distant edges based on distance

    Args:
        G (nx.Graph): full cellular graph of the region
        neighbor_edge_cutoff (float): distance cutoff for neighbor edges.
        # TODO: no more defaults
            # By default we use 55 pixels (~20 um)

    Returns:
        dict: edge properties
    """
    edge_properties = {}
    for i, j in G.edges:
        ci = G.nodes[i]["center_coord"]
        cj = G.nodes[j]["center_coord"]
        dist = np.linalg.norm(np.array(ci) - np.array(cj), ord=2)
        edge_properties[(i, j)] = {
            "distance": dist,
            "edge_type": "neighbor" if dist < neighbor_edge_cutoff else "distant",
        }
    return edge_properties


def merge_cell_dataframes(df1, df2):
    """Merge two cell dataframes on shared rows (cells)"""
    if set(df2["CELL_ID"]) != set(df1["CELL_ID"]):
        warnings.warn("Cell ids in the two dataframes do not match")
    shared_cell_ids = set(df2["CELL_ID"]).intersection(set(df1["CELL_ID"]))
    df1 = df1[df1["CELL_ID"].isin(shared_cell_ids)]
    df1 = df1.merge(df2, on="CELL_ID")
    return df1


def construct_graph_for_region(
    region_id,
    cell_coords_file=None,
    cell_types_file=None,
    cell_biomarker_expression_file=None,
    cell_features_file=None,
    voronoi_file=None,
    graph_source="polygon",
    graph_output=None,
    voronoi_polygon_img_output=None,
    graph_img_output=None,
    figsize=10,
):
    """Construct cellular graph for a region

    Args:
        region_id (str): region id
        cell_coords_file (str): path to csv file containing cell coordinates
        cell_types_file (str): path to csv file containing cell types/annotations
        cell_biomarker_expression_file (str): path to csv file containing cell biomarker expression
        cell_features_file (str): path to csv file containing additional cell features
            Note that features stored in this file can only be numeric and
            will be saved and used as is.
        voronoi_file (str): path to the voronoi coordinates file
        graph_source (str): source of edges in the graph, either "polygon" or "cell"
        graph_output (str): path for saving cellular graph as gpickle
        voronoi_polygon_img_output (str): path for saving voronoi image
        graph_img_output (str): path for saving dot-line graph image
        figsize (int): figure size for plotting

    Returns:
        G (nx.Graph): full cellular graph of the region
    """
    assert cell_coords_file is not None, "cell coordinates must be provided"
    cell_data = load_cell_coords(cell_coords_file)

    if voronoi_file is None:
        # Calculate voronoi polygons based on cell coordinates
        voronoi_polygons = calcualte_voronoi_from_coords(cell_data["X"], cell_data["Y"])
    else:
        # Load voronoi polygons from file
        voronoi_polygons = read_raw_voronoi(voronoi_file)

    if cell_types_file is not None:
        # Load cell types
        cell_types = load_cell_types(cell_types_file)
        cell_data = merge_cell_dataframes(cell_data, cell_types)

    if cell_biomarker_expression_file is not None:
        # Load cell biomarker expression
        cell_expression = load_cell_biomarker_expression(cell_biomarker_expression_file)
        cell_data = merge_cell_dataframes(cell_data, cell_expression)

    if cell_features_file is not None:
        # Load additional cell features
        additional_cell_features = load_cell_features(cell_features_file)
        cell_data = merge_cell_dataframes(cell_data, additional_cell_features)

    if graph_source == "polygon":
        # Build initial cellular graph
        G = build_graph_from_voronoi_polygons(voronoi_polygons)
        # Construct matching between voronoi polygons and cells
        node_to_cell_mapping = build_voronoi_polygon_to_cell_mapping(G, voronoi_polygons, cell_data)
        # Prune graph to contain only voronoi polygons that have corresponding cells
        G = G.subgraph(node_to_cell_mapping.keys())
    elif graph_source == "cell":
        G, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
    else:
        raise ValueError("graph_source must be either 'polygon' or 'cell'")

    # Assign attributes to cellular graph
    G = assign_attributes(G, cell_data, node_to_cell_mapping)
    G.region_id = region_id

    # Visualization of cellular graph
    if voronoi_polygon_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_voronoi_polygons(G)
        plt.axis("scaled")
        plt.savefig(voronoi_polygon_img_output, dpi=300, bbox_inches="tight")
    if graph_img_output is not None:
        plt.clf()
        plt.figure(figsize=(figsize, figsize))
        plot_graph(G)
        plt.axis("scaled")
        plt.savefig(graph_img_output, dpi=300, bbox_inches="tight")

    # Save graph to file
    if graph_output is not None:
        with open(graph_output, "wb") as f:
            pickle.dump(G, f)
    return G


if __name__ == "__main__":
    raw_data_root = "data/voronoi/"
    nx_graph_root = "data/example_dataset/graph"
    fig_save_root = "data/example_dataset/fig"
    os.makedirs(nx_graph_root, exist_ok=True)
    os.makedirs(fig_save_root, exist_ok=True)

    region_ids = sorted(set(f.split(".")[0] for f in os.listdir(raw_data_root)))

    for region_id in region_ids:
        print("Processing %s" % region_id)
        cell_coords_file = os.path.join(raw_data_root, "%s.cell_data.csv" % region_id)
        cell_types_file = os.path.join(raw_data_root, "%s.cell_types.csv" % region_id)
        cell_biomarker_expression_file = os.path.join(raw_data_root, "%s.expression.csv" % region_id)
        cell_features_file = os.path.join(raw_data_root, "%s.cell_features.csv" % region_id)
        voronoi_file = os.path.join(raw_data_root, "%s.json" % region_id)

        voronoi_img_output = os.path.join(fig_save_root, "%s_voronoi.png" % region_id)
        graph_img_output = os.path.join(fig_save_root, "%s_graph.png" % region_id)
        graph_output = os.path.join(nx_graph_root, "%s.gpkl" % region_id)

        if not os.path.exists(graph_output):
            G = construct_graph_for_region(
                region_id,
                cell_coords_file=cell_coords_file,
                cell_types_file=cell_types_file,
                cell_biomarker_expression_file=cell_biomarker_expression_file,
                cell_features_file=cell_features_file,
                voronoi_file=voronoi_file,
                graph_output=graph_output,
                voronoi_polygon_img_output=voronoi_img_output,
                graph_img_output=graph_img_output,
                figsize=10,
            )
