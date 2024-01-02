from cluster_comparison import read_run_result_ann_data
from helpers import ANNOTATION_DICT
# import dash
# import flask

y_data_filter_name = "Liver12Slice12"
y_resolution = 0.7
ann_data = read_run_result_ann_data(y_data_filter_name, y_resolution)
ann_data.obs = ann_data.obs.rename(columns={"sample": "liver_slice"})
ann_data.obs['CELL_TYPE'] = ann_data.obs['leiden_res'].map(ANNOTATION_DICT).tolist()
print(ann_data)


categorical_columns = ann_data.obs.select_dtypes(
    include=["category", "object", "bool"]
).columns.tolist()
numerical_columns = ann_data.obs.select_dtypes(
    include=["float32", "int32", "float64", "int64"]
).columns.tolist()
