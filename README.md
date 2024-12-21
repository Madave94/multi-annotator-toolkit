# Multi Annotator Toolkit

This is a plugin for the FiftyOne framework that allows analysis of multi-annotated data. The plugin is still under 
development. It is associated with our WACV publication [[1]](#1) looking into Label Convergence in Object Recognition
as well as analysis of annotation errors/variations.

## Installation

Install fiftyone [as described](https://github.com/voxel51/fiftyone) on the official website. 

Install this plugin:
```
fiftyone plugins download https://github.com/Madave94/multi-annotator-toolkit
```
Check if plugin is installed:
```
fiftyone plugins list
```
Install plugin dependencies as described in [requirements.txt](requirements.txt) using auto-install:
```
fiftyone plugins requirements @madave94/multi_annotator_toolkit --install
```

## Multi-Annotator-Toolkit Hello World using LVIS

Download the consistency data:
[https://drive.google.com/file/d/1wmZ76Q7D1yP3VVa4kn3DhnWH0tU1Dlmz/view?usp=sharing](https://drive.google.com/file/d/1wmZ76Q7D1yP3VVa4kn3DhnWH0tU1Dlmz/view?usp=sharing)

Unpack the data.

You can copy this example into a python script, **change the data_root variable** and run it:

```
import fiftyone as fo
import fiftyone.operators as foo
print(fo.list_datasets())
name = "LVIS-Multi-Annoated-Subset"
if fo.dataset_exists(name):
    fo.delete_dataset(name)

data_root = "/path/to/downloaded/folder/" # <---- change this
labels_path = data_root + "lvis_v1.0_val_doubly_annos_subset200.json"
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_root + "images",
    labels_path=labels_path,
    name = name,
    extra_attrs = True,
    use_polylines =True
)

load_multi_annotated_data = foo.get_operator("@madave94/multi_annotator_toolkit/load_multi_annotated_data")
await load_multi_annotated_data(dataset, labels_path)
calculate_iaa = foo.get_operator("@madave94/multi_annotator_toolkit/calculate_iaa")
await calculate_iaa(dataset, "bounding box", [0.5, 0.6])
await calculate_iaa(dataset, "polygon", [0.8, 0.9])
session = fo.launch_app(dataset, auto=False)
```

Ensure to change the path to the downloaded and unpacked folder. This example will load the multi-annotated and run a
inter-annotator-agreement calculation.



## Reference

<a id="1">[1]</a> Tschirschwitz David and Rodehorst Volker . "Label Convergence: Defining an Upper Performance Bound in 
Object Recognition through Contradictory Annotations" _Proceedings of the IEEE/CVF Winter Conference on Applications of 
Computer Vision (WACV)_. 2025. [Link](https://arxiv.org/abs/2409.09412)