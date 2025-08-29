import os
import json
from collections import defaultdict
from itertools import combinations
import traceback
import random
import logging
from copy import deepcopy
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List, Union

import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F
import fiftyone.utils.iou as foui
import fiftyone.core.labels as fol
from pymongo.errors import DocumentTooLarge, WriteError

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from kalphacv import reliability_data, krippendorff_alpha

_NO_MATCH_ID = ""
_NO_MATCH_IOU = None
AnnotationList = List[fol.Label]

class LoadMultiAnnotatedData(foo.Operator):
    """
        An operator to load multi-annotated data into a FiftyOne `COCODetectionDataset` and split annotations by `rater_id`.

        This operator addresses the challenge of loading datasets with multiple annotations per image (from different raters)
        into FiftyOne, which does not natively support multi-annotations in a single `COCODetectionDataset`. It provides
        a workaround by processing the dataset to include the necessary fields and splitting the annotations accordingly.

        **Requirements:**

        1. **Annotations**:
           - Each annotation in your dataset must include a `rater_id` field indicating the annotator who provided the annotation.
           - Example annotation with `rater_id`:

             ```json
             {
                 "id": 1,
                 "image_id": 1,
                 "category_id": 3,
                 "bbox": [x, y, width, height],
                 "area": area,
                 "iscrowd": 0,
                 "rater_id": "r1"
             }
             ```

        2. **Images**:
           - Each image in your dataset should have a `rater_list` field, which is a list of all `rater_id`s associated with that image.
           - Example image with `rater_list`:

             ```json
             {
                 "id": 1,
                 "file_name": "image1.jpg",
                 "height": height,
                 "width": width,
                 "rater_list": ["r1", "r2"]
             }
             ```

        3. **Enable `extra_attrs`**:
           - When importing your dataset using `COCODetectionDataset`, you must set `extra_attrs=True` to ensure that
           extra fields like `rater_id` and `rater_list` are loaded into the dataset.

        **Usage:**

        **1. Load Your Dataset with Extra Attributes:**

        ```python
        import fiftyone as fo

        labels_path = "/path/to/your/annotations.json"
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path="/path/to/your/images/",
            labels_path=labels_path,
            name="your_dataset_name",
            extra_attrs=True,  # Important to include extra fields like 'rater_id' and 'rater_list'
            use_polylines = True/False, # select True or False depending on what you want to use, polylines are usually better
            label_types=["detections", "segmentations"], # in case you use an object detection dataset use only "detections"
        )
        ```

        **2. Use the Operator via the SDK (jupyter-notebook example):**

        ```python
        import fiftyone.operators as foo

        load_multi_annotated_data = foo.get_operator("@madave94/multi_annotator_toolkit/load_multi_annotated_data")
        await load_multi_annotated_data(dataset, labels_path)
        ```

        **3. Or Use the Operator via the FiftyOne App UI:**

        - Open the FiftyOne App.
        - Navigate to the Operators panel.
        - Select "Load Multi Annotated Data" from the list of available operators.
        - Provide the path to your annotation file when prompted. (same path as before used for labels_path)

        **Note:** When using the UI function, the progress bar may not be displayed.

        **What the Operator Does:**

        - **Loads `rater_list` into each sample**: Associates each image with its list of raters based on the provided `rater_list` field in the annotations.
        - **Splits annotations by `rater_id`**: For each annotation field (e.g., `detections`, `segmentations`), the operator splits the annotations into separate fields for each rater. For example, annotations from `rater_id` "r1" will be moved to a new field `detections_r1`.
        - **Handles Missing Annotations Gracefully**: If some annotations do not have a `rater_id`, they will remain in the original annotation field.

        **Example Annotation Format:**

        Your annotation JSON file should follow the COCO format with the additional `rater_id` and `rater_list` fields:

        ```json
        {
            "images": [
                {
                    "id": 1,
                    "file_name": "image1.jpg",
                    "height": 1024,
                    "width": 768,
                    "rater_list": ["r1", "r2"]
                },
                // ... more images ...
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 3,
                    "bbox": [100, 200, 50, 80],
                    "area": 4000,
                    "iscrowd": 0,
                    "rater_id": "r1"
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 5,
                    "bbox": [150, 250, 60, 90],
                    "area": 5400,
                    "iscrowd": 0,
                    "rater_id": "r2"
                },
                // ... more annotations ...
            ],
            "categories": [
                {
                    "id": 3,
                    "name": "category_name_1",
                    "supercategory": "supercategory_name"
                },
                // ... more categories ...
            ]
        }
        ```

        **Important Notes:**

        - **Workaround Implementation**: This operator provides a workaround for handling multi-annotated data in FiftyOne, which does not natively support multiple annotations per sample in the same field.
        - **Data Integrity**: Ensure that your annotations and images include the `rater_id` and `rater_list` fields to allow the operator to process them correctly.
        - **Field Naming**: The operator creates new fields in your dataset, such as `detections_r1`, `detections_r2`, etc., for each rater. These fields will contain the annotations specific to each rater.
        - **Processing Limitations**: The operator currently supports splitting annotations in the `detections` and `segmentations` fields. If your dataset uses different fields, you may need to adjust the operator accordingly.
        - **Backup Recommendation**: Since the operator modifies your dataset by reorganizing annotations, it is recommended to backup your dataset before applying the operator, especially if you plan to make further modifications.

        """
    @property
    def config(self):
        return foo.OperatorConfig(
            name="load_multi_annotated_data",
            label="Load Multi Annotated Data",
            description="This loads the multi-annotated meta-data like the rater-list into the samples and splits up the annotations by rater_id.",
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def __call__(self, sample_collection, annos_path, overwrite=False, num_workers=False, delegate=False):
        ctx = dict(view=sample_collection.view())
        params = dict(annos_path=annos_path, overwrite=overwrite, num_workers=num_workers, delegate=delegate, api_call=True)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            return None
        # --- for SDK call ---

        inputs = types.Object()

        inputs.str("annotation_path_instructions",
                   default="Provide the path to the annotation file (same as the one used for loading the dataset previously.",
                   view=types.MarkdownView(read_only=True))

        # Create an explorer that allows the user to choose a JSON file
        file_explorer = types.FileExplorerView(
            button_label="Choose a JSON file...",
            choose_button_label="Select",
            choose_dir=False
        )

        # Define a types.File property with the file explorer
        inputs.file(
            "annos_path",
            required=True,
            label="Annotation file",
            description="Choose an annotation file",
            view=file_explorer
        )

        annos_path = ctx.params.get("annos_path", None)

        if annos_path is None:
            return types.Property(inputs)  # Wait for user input

        # Extract the file path from the annos_path dictionary
        if isinstance(annos_path, dict):
            file_path = annos_path.get('absolute_path') or annos_path.get('path')
        else:
            file_path = annos_path

        # check first if the file path is json and than check if it exists
        if not file_path.lower().endswith('.json'):
            prop = inputs.get_property('annos_path')
            prop.invalid = True
            prop.error_message = "Please select a file with a .json extension."
            return types.Property(inputs)

        if not os.path.isfile(file_path):
            prop = inputs.get_property('annos_path')
            prop.invalid = True
            prop.error_message = f"The file '{file_path}' does not exist."
            return types.Property(inputs)

        return types.Property(inputs)

    def execute(self, ctx):
        try:
            annos_path = ctx.params.get("annos_path")
            if isinstance(annos_path, dict):
                file_path = annos_path.get('absolute_path') or annos_path.get('path')
            else:
                file_path = annos_path

            # Proceed with your logic using file_path
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Create a mapping from base file names to rater_list
            file_name_to_rater_list = {
                os.path.basename(image["file_name"]): image["rater_list"] for image in data["images"]
            }

            # Access the dataset
            dataset = ctx.dataset

            # Ensure 'rater_list' field is defined as ListField of StringField
            if 'rater_list' not in dataset.get_field_schema():
                dataset.add_sample_field('rater_list', fo.ListField, subfield=fo.StringField)

            # Create a mapping from sample IDs to rater_list
            id_to_rater_list = {}

            # Iterate over the samples
            for sample in dataset:
                # Extract the base file name from the sample's filepath
                sample_file_name = os.path.basename(sample.filepath)

                # Get the rater_list for this sample, if it exists
                try:
                    rater_list = file_name_to_rater_list[sample_file_name]
                    id_to_rater_list[sample.id] = rater_list
                except KeyError:
                    raise ValueError(
                        f"Data Inconsistency: The image file '{sample_file_name}' "
                        f"does not contain the required 'rater_list' attribute."
                        f"Ensure that the annotation file is correctly formatted."
                    )

            # Bulk update the rater_list field
            if id_to_rater_list:
                # Specify key_field="_id" to match sample IDs
                dataset.set_values("rater_list", id_to_rater_list, key_field="_id")
                num_updated = len(id_to_rater_list)
            else:
                num_updated = 0

            # Create an index on 'rater_list' for faster queries
            if "rater_list" not in dataset.list_indexes():
                dataset.create_index("rater_list")

            # Check which annotation fields exist
            field_schema = dataset.get_field_schema()
            messages = []
            loading_success = False

            #
            #  This should include 3 cases
            #  1) detections and segmentations are available
            #  2) only detections are loaded
            #  3) only segmentations are loaded
            #
            #  -> for the segmentations there is a difference between loading polylines or masks
            #
            # The approach is to check if case 2 or 3 exist which means the field is stored into ground_truth instead of
            # segmentations or detections. We find out the type and map it to the correct type. At this point the regular
            # process is followed.

            # check for ground_truth type and rename it
            if "ground_truth" in field_schema:
                for sample in dataset:
                    gt = sample["ground_truth"]
                    if gt == None:
                        continue
                    else:
                        if hasattr(gt, "detections"):
                            if "mask" in gt.detections[0]:
                                dataset.rename_sample_field("ground_truth", "segmentations")
                                print("Rename ground_truth to segmentations..")
                            else:
                                dataset.rename_sample_field("ground_truth", "detections")
                                print("Rename ground_truth to detections..")
                        elif hasattr(gt, "polylines"):
                            dataset.rename_sample_field("ground_truth", "segmentations")
                            print("Rename ground_truth to segmentations..")
                        else:
                            raise Exception("Could not identify annotation type.")
                    field_schema = dataset.get_field_schema()
                    break

            ann_types = []

            if 'detections' in field_schema:
                ann_types.append("bounding box")
                detection_counts = split_annotations_by_rater(dataset, 'detections', ctx=ctx)
                messages.append(
                    f"Detections - Total: {detection_counts['total_annotations']},    \n"
                    f"Moved: {detection_counts['annotations_moved']},    \n"
                    f"Unassigned: {detection_counts['annotations_unassigned']}.    \n"
                )
                loading_success = True

            if 'segmentations' in field_schema:
                ann_types.append(return_segmentation_type(dataset))
                segmentation_counts = split_annotations_by_rater(dataset, 'segmentations')
                messages.append(
                    f"Segmentations - Total: {segmentation_counts['total_annotations']},    \n"
                    f"Moved: {segmentation_counts['annotations_moved']},    \n"
                    f"Unassigned: {segmentation_counts['annotations_unassigned']}.    \n"
                )
                loading_success = True

            dataset.info["ann_types"] = ann_types
            dataset.save()

            # **Join the messages into a single string**
            if loading_success:
                messages = [f"Successfully loaded multi-annotations for {num_updated} out of {len(dataset)} samples.\n    "] + messages
            else:
                messages = ["Loading unsuccessful.\n    "]
            message_str = "\n".join(messages)

            print(message_str)

            return {
                "message": message_str,
                "num_updated": num_updated,
                "num_samples": len(dataset),
            }
        except Exception as e:
            error_details = traceback.format_exc()
            error_message = (
                f"❌ Operator failed!\n\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Details: {e}\n\n"
                f"Full Traceback:\n-----------------\n{error_details}"
            )

            print(error_message)

            return {"message": error_message}

    def resolve_output(self, ctx):
        outputs = types.Object()

        # Display the message as a notice
        outputs.view(
            "message",
            types.Notice(label=ctx.results.get("message", "")),
        )
        print(ctx.results.get("message", ""))

        return types.Property(outputs)

# utility function to process annotations
def split_annotations_by_rater(dataset, source_field: str, field_prefix: str =None, ctx=None) -> dict:
    """
    Splits annotations in the source_field into per-rater fields based on 'rater_id'.

    Parameters:
    - dataset: the FiftyOne dataset
    - source_field: the name of the source field to process ('detections' or 'segmentations')
    - field_prefix: prefix for the per-rater fields (default: source_field + '_')
    """
    if field_prefix is None:
        field_prefix = source_field + '_'

    # Counters for reporting
    total_annotations = 0
    annotations_moved = 0
    annotations_unassigned = 0
    samples_processed = 0

    # Get the source field's type from the dataset schema once
    source_field_doc_type = dataset.get_field(source_field).document_type

    all_raters = set()
    for rater_list in dataset.values("rater_list"):
        all_raters.update(rater_list)

    # Ensure field exists in dataset schema -> executed only once for the entire dataset per field
    # this needs to be done so that empty images are still added
    for rater_id in all_raters:
        sanitized_rater_id = _sanitize_for_field_name(rater_id).replace('.', '_')
        field_name = f"{field_prefix}{sanitized_rater_id.replace('.', '_')}"
        if not dataset.has_sample_field(field_name):
            dataset.add_sample_field(
                field_name,
                fo.EmbeddedDocumentField,
                embedded_doc_type=source_field_doc_type
            )

    # Process each sample
    for sample in tqdm(dataset, desc=f"Processing {source_field}"):
        original_rater_list = sample.get_field("rater_list")
        if not original_rater_list:
            raise Exception(f"Missing 'rater_list' for {sample.filepath}.")
        sanitized_rater_list = [_sanitize_for_field_name(r) for r in original_rater_list]
        sample.rater_list = sanitized_rater_list

        # Initialize per-rater annotations dict
        annotations_by_rater = {sanitized_id: [] for sanitized_id in sanitized_rater_list}
        unassigned_annotations = []

        annotations: fol.Label = sample.get_field(source_field)

        # Determine the attribute to access based on field type
        if annotations is not None:
            if isinstance(annotations, fo.Detections):
                annotations_list: AnnotationList = annotations.detections
            elif isinstance(annotations, fo.Polylines):
                annotations_list: AnnotationList = annotations.polylines
            else:
                raise TypeError(f"Unsupported label type {type(annotations)}.")

            # Process annotations
            for annotation in annotations_list:
                original_rater_id = annotation.get_field('rater_id')
                if not original_rater_id:
                    raise ValueError(f"Missing 'rater_id' for annotation in {sample.filepath}")
                sanitized_rater_id = _sanitize_for_field_name(original_rater_id)
                annotation.rater_id = sanitized_rater_id
                total_annotations += 1

                if sanitized_rater_id in annotations_by_rater:
                    annotations_by_rater[sanitized_rater_id].append(annotation)
                    annotations_moved += 1
                else:
                    unassigned_annotations.append(annotation)
                    annotations_unassigned += 1

        # Assign per-rater annotations to new fields
        for rater_id in sanitized_rater_list:
            ann_list = annotations_by_rater.get(rater_id, [])
            field_name = f"{field_prefix}{rater_id}"
            if ann_list:
                if isinstance(annotations, fo.Detections):
                    sample[field_name] = fo.Detections(detections=ann_list)
                elif isinstance(annotations, fo.Polylines):
                    sample[field_name] = fo.Polylines(polylines=ann_list)
                else:
                    raise Exception("Invalid annotations type processed. Should be detections or polylines.")
            else:
                sample[field_name] = source_field_doc_type()

        # Handle unassigned annotations
        if unassigned_annotations:
            # Keep them in the source field
            if isinstance(annotations, fo.Detections):
                sample[source_field] = fo.Detections(detections=unassigned_annotations)
            elif isinstance(annotations, fo.Polylines):
                sample[source_field] = fo.Polylines(polylines=unassigned_annotations)
            else:
                raise Exception("Invalid annotations type processed. Should be detections or polylines.")
        else:
            # No unassigned annotations; clear the source field
            sample.clear_field(source_field)

        # Save the sample
        sample.save()
        samples_processed += 1

    # Optionally, remove the source field from the dataset schema if empty
    if dataset.match(F(source_field).exp()).count() == 0:
        dataset.delete_sample_field(source_field)

    # Return counts for reporting
    return {
        'total_annotations': total_annotations,
        'annotations_moved': annotations_moved,
        'annotations_unassigned': annotations_unassigned,
        'samples_processed': samples_processed
    }

def return_segmentation_type(dataset):
    for sample in dataset:
        annotations = sample.get_field("segmentations")
        if annotations is None:
            continue
        # Determine the attribute to access based on field type
        if isinstance(annotations, fo.Detections):
            return "mask"
        elif isinstance(annotations, fo.Polylines):
            return "polygon"
        else:
            raise Exception("Invalid annotations type processed. Should be detections or polylines.")

class CalculateIaa(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="calculate_iaa",
            label="Calculate IAA",
            description="Calculates the Inter-Annotator-Agreement",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            dynamic=True,
        )

    def __call__(self, sample_collection, annotation_type, iou_thresholds, run_sampling=False, subset_n=None,
                 sampling_k=None, random_seed_s=None, delegate=False):
        ctx = dict(view=sample_collection.view())
        # set default parameters for sampling procedure if sampling is activated
        if run_sampling:
            subset_n = subset_n if subset_n else int(len(sample_collection) * 0.1)
            sampling_k = sampling_k if sampling_k else 1000
            random_seed_s = random_seed_s if random_seed_s else 42
        params = dict(annotation_type=annotation_type, iou_thresholds=iou_thresholds, run_sampling=run_sampling,
                      subset_n=subset_n, sampling_k=sampling_k, random_seed_s=random_seed_s, delegate=delegate, api_call=True)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # Check available annotation types (bbox, polygon, mask)
        available_types = check_available_annotation_types(ctx)
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            assert ctx.params.get("annotation_type") in available_types, \
                "Annotation type {} not in {}.".format(ctx.params.get("annotation_type"), available_types)
            return None
        # --- for SDK call ---

        inputs = types.Object()

        inputs.md("###### Options for calculating inter annotator agreement", name="mk1")

        # Create checkboxes for available annotation types
        annotation_types_radio_group = types.RadioGroup()
        for annotation_type in available_types:
            annotation_types_radio_group.add_choice(annotation_type, label=annotation_type)

        inputs.enum(
            "annotation_type",
            annotation_types_radio_group.values(),
            label="Annotation Type",
            description="Select the annotation type to include in the analysis:",
            types=types.RadioView(),
            default=list(available_types)[0],
        )

        inputs.list(
            "iou_thresholds",
            types.Number(min=0.01, max=0.99, float=True),
            label="IoU Thresholds",
            description="Enter IoU thresholds. Values should range between 0.01 and 0.99.",
        )

        inputs.bool(
            "run_sampling",
            default=False,
            label="Run sampling",
            description="Run sampling procedure to produce confidence interval when determining convergence threshold."
        )

        run_sampling = ctx.params.get("run_sampling")

        if run_sampling:

            dataset = ctx.dataset

            inputs.md("""
                         Selecting "Run sampling" will still first run IAA on all samples but than, the sampling procedure 
                         applies **bootstrapping**, where `k` replicates of size `n` are drawn from the dataset `N` with 
                         replacement, using a fixed random seed (`s`) for reproducibility.   
                         Please consider deligating this operation, since for larger k or n the processing might take 
                         multiple hours.
                      """, name="mk2")

            inputs.int(
                "subset_n",
                label="Subset Size (n)",
                description="Size of the dataset sample replicate n, with maximum size: dataset size - 1",
                default=int(len(dataset)*0.1),
                min=1,
                max=len(dataset)-1,
                required=True
            )

            inputs.int(
                "sampling_k",
                label="Number of Samples (k)",
                description="The number of times the sampling procedure is repeated.",
                default=1000,
                min=2,
                required=True
            )

            inputs.int(
                "random_seed_s",
                label="Random Seed (s)",
                description="Select a random seed used to sample from the dataset. Available for reproduction purposes.",
                default=42,
                min=0,
                required=True
            )


        # Add execution mode (if applicable to your use case)
        _execution_mode(ctx, inputs)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        # Access the dataset
        dataset = ctx.dataset

        ann_type = ctx.params.get("annotation_type")

        iou_thresholds = ctx.params.get("iou_thresholds")

        # Add the field to the dataset if it does not already exist
        if not dataset.has_sample_field("iaa"):
            dataset.add_sample_field(
                "iaa",
                fo.DictField,
                subfield=fo.FloatField
            )

        if "iaa_analyzed" not in dataset.info:
            dataset.info["iaa_analyzed"] = []
            dataset.save()
        for iou in iou_thresholds:
            iou_str = str(iou).replace(".", ",")
            key = ann_type + "-" + iou_str
            if key not in dataset.info["iaa_analyzed"]:
                dataset.info["iaa_analyzed"].append(key)
                dataset.save()

        alphas = defaultdict(list)
        for sample in tqdm(dataset, "Calculating IAA for samples"):
            image_name = sample.get_field("filepath").split("/")[-1]
            raters_by_image = sample.get_field("rater_list")
            if raters_by_image is None:
                continue  # Skip samples without raters

            size_for_image = (sample.metadata.height, sample.metadata.width)
            annotations_by_image = []

            for rater_id in raters_by_image:
                if ann_type == "bounding box":
                    ann_field = f"detections_{rater_id}"
                    element_field = "detections"
                elif ann_type == "mask":
                    ann_field = f"segmentations_{rater_id}"
                    element_field = "detections"
                elif ann_type == "polygon":
                    ann_field = f"segmentations_{rater_id}"
                    element_field = "polylines"
                else:
                    continue

                if not sample.has_field(ann_field):
                    continue
                annotations = sample.get_field(ann_field)
                if annotations is None:
                    continue

                for annotation in getattr(annotations, element_field, []):
                    segmentation = None
                    if ann_type == "polygon":
                        segmentation = [
                            [point for sublist in shape for point in sublist]
                            for shape in annotation["points"]
                        ]
                    elif ann_type == "mask":
                        segmentation = annotation["mask"]

                    annotations_by_image.append({
                        "bbox": annotation["bounding_box"] if "bounding_box" in annotation else None,
                        "category_id": annotation["label"],
                        "rater": rater_id,
                        "segmentation": segmentation,
                    })

            # Prepare reliability data
            rel_data = reliability_data.ReliabilityData(
                image_name, annotations_by_image, raters_by_image, size_for_image
            )
            for iou_threshold in iou_thresholds:
                coincidence_matrix = rel_data.run("bbox" if "bounding box" == ann_type else "segm",
                                                  iou_threshold)
                alpha = krippendorff_alpha.calculate_alpha(coincidence_matrix)

                iaa_dict = sample["iaa"]
                if iaa_dict is None:
                    iaa_dict = {}
                iou_str = str(iou_threshold).replace(".", ",")
                iaa_dict[ann_type + "-" + iou_str] = alpha
                sample["iaa"] = iaa_dict
                sample.save()
                alphas[str(iou_threshold)].append(alpha)

        run_sampling = ctx.params.get("run_sampling")
        if run_sampling:
            subset_n = ctx.params.get("subset_n", int(len(dataset)*0.1))
            sampling_k = ctx.params.get("sampling_k", 1000)
            random_seed_s = ctx.params.get("random_seed_s", 42)
            if "iaa_sampled" not in dataset.info:
                dataset.info["iaa_sampled"] = {}
            iaas = dataset.info["iaa_sampled"]
            random.seed(random_seed_s)
            for idx in range(sampling_k):
                # sample iaa value per threshold
                indices = random.sample(range(len(dataset)), subset_n)
                for iou_threshold in iou_thresholds:
                    iaa_values = [alphas[str(iou_threshold)][i] for i in indices]
                    iaas[f"{ann_type}_{iou_threshold}_{random_seed_s}_{subset_n}_{idx}"] = sum(iaa_values) / len(iaa_values)

            del dataset.info["iaa_sampled"]
            dataset.info["iaa_sampled"] = iaas
            dataset.save()

        message = "Mean K-Alpha for:    \n"
        for iou_threshold in iou_thresholds:
            u_k_alpha = sum(alphas[str(iou_threshold)]) / len(alphas[str(iou_threshold)])
            message += f"\tIoU {iou_threshold} on {ann_type}: {u_k_alpha}    \n"

        # Include a message for sampling
        if run_sampling:
            message += f"\tSampling completed with {sampling_k} samples of size {subset_n} using random seed {random_seed_s}.    \n"

        print(message)

        ctx.ops.open_panel("iaa_panel")

        return {"message": message}

    def resolve_output(self, ctx):
        outputs = types.Object()

        # Display the message as a notice
        outputs.view(
            "message",
            types.Notice(label=ctx.results.get("message", "")),
        )
        print(ctx.results.get("message", ""))

        return types.Property(outputs)


class IAAPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="iaa_panel",
            label="IAA Panel",
            allow_multiple=False,
            surfaces="grid",
            help_markdown="A panel to filter IAA values in the views and show summary statistics.",
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def on_load(self, ctx):
        # load initial values
        iaa_list = ctx.dataset.info["iaa_analyzed"]

        iaa_dict = defaultdict(list)
        for iaa in iaa_list:
            ann_type, iou = iaa.split("-")
            iaa_dict[ann_type].append(iou)

        # set default values
        ctx.panel.state.iaa_dict = iaa_dict
        if ctx.panel.state.ann_type_selection is None:
            ctx.panel.state.ann_type_selection = list(iaa_dict.keys())[0]
        if ctx.panel.state.iou_selection is None:
            ctx.panel.state.iou_selection = iaa_dict[ctx.panel.state.ann_type_selection][0]

        self.apply(ctx)

        ctx.ops.split_panel("iaa_panel", layout="horizontal")

    def apply(self, ctx):
        # store values
        ann_type_selection = ctx.panel.state.ann_type_selection
        iou_selection = ctx.panel.state.iou_selection
        iaa_dict = ctx.panel.state.iaa_dict

        # clean values
        ctx.ops.clear_panel_state()

        # initialize default min/max values
        ctx.panel.state.max_value = 1.0
        ctx.panel.state.min_value = -1.0
        ctx.panel.set_state("v_stack.double_slider", [-1.0,1.0])

        # set states
        ctx.panel.state.ann_type_selection = ann_type_selection
        ctx.panel.state.iou_selection = iou_selection
        ctx.panel.state.iaa_dict = iaa_dict
        ctx.panel.state.mean_msg = "Waiting for calculation ..."

        # run computation and plotting
        ctx.panel.state.plot_title = "Inter-Annotat-Agreement: {} {}".format(
            ctx.panel.state.ann_type_selection,
            ctx.panel.state.iou_selection)
        self.select_values(ctx)
        self.set_histogram_values(ctx)

        ctx.ops.clear_view()

    def change_ann_type(self, ctx):
        ctx.panel.state.ann_type_selection = ctx.params["value"]

    def change_iou_value(self, ctx):
        ctx.panel.state.iou_selection = ctx.params["value"]

    def select_values(self, ctx):
        selected_values = []
        unselected_values = []
        for sample in ctx.dataset:
            value = (sample["iaa"][ctx.panel.state.ann_type_selection + "-" + ctx.panel.state.iou_selection])
            if value <= ctx.panel.state.max_value and value >= ctx.panel.state.min_value:
                selected_values.append(value)
            else:
                unselected_values.append(value)
        ctx.panel.state.selected_values = selected_values
        ctx.panel.state.unselected_values = unselected_values

    def set_histogram_values(self, ctx):
        # Reset the histogram before setting new data
        ctx.panel.state.mean_msg = "Mean IAA for {} samples using {} annotations wih iou-threshold {}: **{:.3f}**".format(
            len(ctx.panel.state.selected_values),
            ctx.panel.state.ann_type_selection,
            ctx.panel.state.iou_selection,
            sum(ctx.panel.state.selected_values) / len(ctx.panel.state.selected_values)
        )
        ctx.ops.clear_panel_data()
        ctx.panel.set_data("v_stack.histogram",  [{"name": "Selected Values",
                                    "x": ctx.panel.state.selected_values,
                                    "type": "histogram",
                                    "marker": {"color": "#FF6D05"}, # gray #808080
                                    "xbins": {"start": -1.0, "end": 1.0001, "size": 0.1},
                                    },
                                    {"name": "other Values",
                                    "x": ctx.panel.state.unselected_values,
                                    "type": "histogram",
                                    "marker": {"color": "#808080"},  # gray #808080
                                    "xbins": {"start": -1.0, "end": 1.0001, "size": 0.1},
                                   }]
                           )

    def on_histogram_click(self, ctx):
        bin_range = ctx.params.get("range")
        min_value = bin_range[0]
        max_value = bin_range[1]

        ann_type = ctx.panel.state.ann_type_selection
        iou_value = ctx.panel.state.iou_selection
        field_name = "iaa.{}-{}".format(ann_type, iou_value)

        ctx.panel.state.min_value = min_value
        ctx.panel.state.max_value = max_value
        self.select_values(ctx)
        self.set_histogram_values(ctx)

        view = ctx.dataset.match((F(field_name) >= min_value) & (F(field_name) <= max_value))

        if view is not None:
            ctx.ops.set_view(view=view)

    def slider_change(self,ctx):
        bin_range = ctx.params.get("value")
        min_value = bin_range[0]
        max_value = bin_range[1]

        ann_type = ctx.panel.state.ann_type_selection
        iou_value = ctx.panel.state.iou_selection
        field_name = "iaa.{}-{}".format(ann_type, iou_value)

        ctx.panel.state.min_value = min_value
        ctx.panel.state.max_value = max_value
        self.select_values(ctx)
        self.set_histogram_values(ctx)

        view = ctx.dataset.match((F(field_name) >= min_value) & (F(field_name) <= max_value))

        if view is not None:
            ctx.ops.set_view(view=view)

    def render(self, ctx):
        panel = types.Object()
        v_stack = panel.v_stack("v_stack", align_x="center", align_y="center", width=100, gap=2)

        h_stack = v_stack.h_stack("h_stack", align_x="center", align_y="center", gap=5)

        dropdown_ann_type = types.DropdownView()
        for ann_type in ctx.panel.state.iaa_dict.keys():
            dropdown_ann_type.add_choice(ann_type, label=ann_type)

        h_stack.view(
            "dropdown_ann_type",
            view=dropdown_ann_type,
            label="Annotation Type",
            on_change=self.change_ann_type
        )

        dropdown_iou_value = types.DropdownView()
        for iou_type in ctx.panel.state.iaa_dict[ctx.panel.state.ann_type_selection]:
            dropdown_iou_value.add_choice(iou_type, label=iou_type)

        h_stack.view(
            "dropdown_iou_value",
            view=dropdown_iou_value,
            label="IoU Threshold",
            on_change=self.change_iou_value
       )

        h_stack.btn(
            "load",
            label="load/reset",
            on_click=self.apply
        )


        v_stack.plot(
            "histogram",
            layout={
                "title": {
                    "text": ctx.panel.state.plot_title,
                    "automargin": True,
                },
                "xaxis": {"title": "K-Alpha"},
                "yaxis": {"title": "Count"},
                "bargap": 0.05,
                "autosize": True,  # Enable autosizing for responsive behavior
                "responsive": True,
                "dragmode": "select",
                "selectdirection": "h",
                "showlegend": True,
                "legend": {
                    "x": 0.5,
                    "y": -0.2,
                    "xanchor": "center",
                    "orientation": "h",
                    "bgcolor": "rgba(0, 0, 0, 0)",  # Transparent background for the legend
                },
                "barmode": "overlay",  # Overlay the two histograms
                "plot_bgcolor": "rgba(0, 0, 0, 0)",  # Transparent background for the plotting area
                "paper_bgcolor": "rgba(0, 0, 0, 0)",  # Transparent background for the entire layout
                "colorway": ["grey", "#FF6D04", "blue"],
            },
            config={
                "scrollZoom": False,
                "displayModeBar": False,
                "responsive": True,
            },
            #on_click=self.on_histogram_click,
            height=75,
        )

        v_stack.view(
            "double_slider",
            on_change=self.slider_change,
            view=types.SliderView(
                value_precision=3,
                variant="withInputs",
                min=-1.0,  # Fixed range min
                max=1.0,  # Fixed range max
                label="Threshold",
                value_label_display="auto",
            ),
        )

        v_stack_small = v_stack.v_stack("v_stack_small", align_x="center", align_y="center", width=75)

        v_stack_small.md(ctx.panel.state.mean_msg)

        return types.Property(
            panel,
            view=types.GridView(
                align_x="center",
                align_y="center",
                #orientation="vertical",
                height=100,
                width=100,
                gap=2,
                padding=0,
            )
        )

class CalculatemmAP(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="calculate_mmap",
            label="Calculate modified mean average precision (mmAP)",
            description="Calculates the modified mean Average Precision for multi-annotated dataset with a maximum of 2 annotators per sample",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            dynamic=True,
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def __call__(self, sample_collection, annotation_type, iou_thresholds, dataset_scope, subset_n=None, sampling_k=None,
                 random_seed_s=None, delegate=False):
        ctx = dict(view=sample_collection.view())
        # set default parameters for sampling procedure if sampling is activated
        if dataset_scope == "Partial":
            subset_n = subset_n if subset_n else int(len(sample_collection) * 0.1)
            sampling_k = sampling_k if sampling_k else 1000
            random_seed_s = random_seed_s if random_seed_s else 42
        params = dict(annotation_type=annotation_type, iou_thresholds=iou_thresholds, dataset_scope=dataset_scope,
                      subset_n=subset_n, sampling_k=sampling_k, random_seed_s=random_seed_s, delegate=delegate, api_call=True)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # Check available annotation types (bbox, polygon, mask)
        available_types = check_available_annotation_types(ctx)
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            assert ctx.params.get("annotation_type") in available_types, \
                "Annotation type {} not in {}.".format(ctx.params.get("annotation_type"), available_types)
            return None
        # --- for SDK call ---
        inputs = types.Object()

        inputs.md("###### Options for calculating modified mean Average Precision", name="mk1")

        dataset = ctx.dataset

        # Create checkboxes for available annotation types
        annotation_types_radio_group = types.RadioGroup()
        for annotation_type in available_types:
            annotation_types_radio_group.add_choice(annotation_type, label=annotation_type)

        inputs.enum(
            "annotation_type",
            annotation_types_radio_group.values(),
            label="Annotation Type",
            description="Select the annotation type to include in the analysis:",
            types=types.RadioView(),
            default=list(available_types)[0],
        )

        inputs.list(
            "iou_thresholds",
            types.Number(min=0.01, max=0.99, float=True),
            label="IoU Thresholds",
            description="Enter IoU thresholds. Values should range between 0.01 and 0.99.",
        )

        dataset_scope = ["Full", "Partial"]
        dataset_scope_group = types.RadioGroup()

        for choice in dataset_scope:
            dataset_scope_group.add_choice(choice, label=choice)

        inputs.enum(
            "dataset_scope",
            dataset_scope_group.values(),
            label="Dataset Scope",
            description="Select if you want to analyze the full dataset or run sampling from the dataset. Sampling "
                        "allows you to run multiple evaluations to evaluate a confidence interval.",
            view=types.RadioView(),
            default=dataset_scope[0]
        )

        dataset_scope_choice = ctx.params.get("dataset_scope", None)

        if dataset_scope_choice == "Partial":

            inputs.md("""
                         For the selected **Partial** dataset scope, the sampling procedure applies **bootstrapping**, 
                         where `k` replicates of size `n` are drawn from the dataset `N` with replacement, using a fixed 
                         random seed (`s`) for reproducibility.   
                         Please consider deligating this operation, since for larger k or n the processing might take 
                         multiple hours.
                      """, name="mk2")

            inputs.int(
                "subset_n",
                label="Subset Size (n)",
                description="Size of the dataset sample replicate n, with maximum size: dataset size - 1",
                default=int(len(dataset)*0.1),
                min=1,
                max=len(dataset)-1,
                required=True
            )

            inputs.int(
                "sampling_k",
                label="Number of Samples (k)",
                description="The number of times the sampling procedure is repeated.",
                default=1000,
                min=2,
                required=True
            )

            inputs.int(
                "random_seed_s",
                label="Random Seed (s)",
                description="Select a random seed used to sample from the dataset. Available for reproduction purposes.",
                default=42,
                min=0,
                required=True
            )

        _execution_mode(ctx, inputs)
        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        dataset = ctx.dataset

        ann_type = ctx.params.get("annotation_type")
        iou_thresholds = ctx.params.get("iou_thresholds")
        dataset_scope_choice = ctx.params.get("dataset_scope")
        subset_n = ctx.params.get("subset_n", None)
        sampling_k = ctx.params.get("sampling_k", None)
        random_seed_s = ctx.params.get("random_seed_s", None)

        if "mmAPs" not in dataset.info:
            dataset.info["mmAPs"] = {}

        mmaps = dataset.info["mmAPs"]

        categories = [{"id": idx, "name": cls_name} for idx, cls_name in enumerate(dataset.default_classes)]
        annotations_dict = {"images": [], "0": [], "1": []}

        annotation_id = 0
        for image_idx, sample in enumerate(tqdm(dataset, desc="Pre-Processing samples")):
            raters_by_image = sample.get_field("rater_list")
            if len(raters_by_image) > 2:
                message = "Error: For cases of 3 or more annotator the implementation on mmAP does not work.    \n" \
                          "Operation cancelled.    \n"
                print(message)
                return {"message": message}

            height, width = sample.metadata.height, sample.metadata.width
            filename = sample.filename
            annotations_dict["images"].append({
                "id": image_idx,
                "height": height,
                "width": width,
                "file_name": filename
            })
            for annotator_idx, rater_id in enumerate(raters_by_image):
                if ann_type == "bounding box":
                    ann_field = f"detections_{rater_id}"
                    element_field = "detections"
                elif ann_type == "mask":
                    ann_field = f"segmentations_{rater_id}"
                    element_field = "detections"
                elif ann_type == "polygon":
                    ann_field = f"segmentations_{rater_id}"
                    element_field = "polylines"
                else:
                    continue
                annotations = sample.get_field(ann_field)
                if annotations is None:
                    continue
                for annotation in getattr(annotations, element_field, []):
                    # populate annotation dict
                    bbox, segm, area = None, None, None
                    # conditional logic to figure out where goes what
                    if ann_type == "bounding box":
                        bbox = list(annotation["bounding_box"])
                        area = calculate_bbox_area(bbox, width, height)
                    elif ann_type == "mask":
                        bbox = list(annotation["bounding_box"])
                        mask = place_mask_in_image(annotation["mask"], bbox, (height, width))
                        segm = coco_mask.encode(np.asfortranarray(mask)) # this is a RLE
                        area = coco_mask.area(segm)
                    elif ann_type == "polygon":
                        segm = [
                            [
                                point * width if i % 2 == 0 else point * height
                                for sublist in shape
                                for i, point in enumerate(sublist)
                            ]
                            for shape in annotation["points"]
                        ]
                        bbox = get_bbox_from_polygon(segm)
                        area = bbox[2] * bbox[3]
                    else:
                        continue

                    annotations_dict[str(annotator_idx)].append({
                        "id": annotation_id,
                        "image_id": image_idx,
                        "category_id": dataset.default_classes.index(annotation["label"]),
                        "bbox": bbox,
                        "segmentation": segm,
                        "area": area,
                        "iscrowd": annotation["iscrowd"],
                        "score": 0.99
                    })
                    annotation_id += 1

        if ann_type == "bounding box":
            annType = "bbox"
        elif ann_type == "mask" or ann_type == "polygon":
            annType = "segm"
        else:
            raise Exception(f"Annotation type {ann_type} does not exist.")
        # check the sampling
        if dataset_scope_choice == "Full":
            ann_dict_0 = {"annotations": annotations_dict["0"], "images": annotations_dict["images"], "categories": categories}
            ann_dict_1 = {"annotations": annotations_dict["1"], "images": annotations_dict["images"], "categories": categories}
            mmap_dict = calc_mmap(ann_dict_0, ann_dict_1, annType, iou_thresholds)
            for thrs in iou_thresholds:
                mmaps[f"{ann_type}_{thrs}_None_all_0"] = mmap_dict[str(thrs)]
        elif dataset_scope_choice == "Partial":
            random.seed(random_seed_s)
            for idx in tqdm(range(sampling_k), desc="Running sampling"):
                sampled_images = random.sample(annotations_dict["images"], subset_n)
                image_ids = [image["id"] for image in sampled_images]
                ann_0 = [annotation for annotation in annotations_dict["0"] if annotation["image_id"] in image_ids]
                ann_1 = [annotation for annotation in annotations_dict["1"] if annotation["image_id"] in image_ids]
                ann_dict_0 = {"annotations": ann_0, "images": sampled_images, "categories": categories}
                ann_dict_1 = {"annotations": ann_1, "images": sampled_images, "categories": categories}
                mmap_dict = calc_mmap(ann_dict_0, ann_dict_1, annType, iou_thresholds)
                for thrs in iou_thresholds:
                    mmaps[f"{ann_type}_{thrs}_{random_seed_s}_{subset_n}_{idx}"] = mmap_dict[str(thrs)]
        else:
            raise Exception(f"Invalid dataset scope {dataset_scope_choice}.")

        del dataset.info["mmAPs"]
        dataset.info["mmAPs"] = mmaps
        dataset.save()

        message = "Modified Mean Average Precision (mmAP) for:    \n"
        for iou_threshold in iou_thresholds:
            if dataset_scope_choice == "Full":
                mmap = mmaps[f"{ann_type}_{iou_threshold}_None_all_0"]
            elif dataset_scope_choice == "Partial":
                list_mmap = []
                for idx in range(sampling_k):
                    list_mmap.append(mmaps[f"{ann_type}_{iou_threshold}_{random_seed_s}_{subset_n}_{idx}"])
                mmap = sum(list_mmap) / len(list_mmap)
            else:
                raise Exception(f"Invalid dataset scope {dataset_scope_choice}.")
            message += f"IoU {iou_threshold} on {ann_type}: {mmap}    \n"

        # Include a message for sampling
        if dataset_scope_choice == "Partial":
            message += f"\nSampling completed with {sampling_k} samples of size {subset_n} using random seed {random_seed_s}.    \n"

        print(message)

        return {"message": message}

    def resolve_output(self, ctx):
        outputs = types.Object()

        # Display the message as a notice
        outputs.view(
            "message",
            types.Notice(label=ctx.results.get("message", "")),
        )

        return types.Property(outputs)

def calc_mmap(dict_a, dict_b, ann_type, iou_thresholds):
    with suppress_output():
        cocoGt_a = COCO()
        cocoGt_b = COCO()
        cocoGt_a.dataset = deepcopy(dict_a)
        cocoGt_b.dataset = deepcopy(dict_b)
        cocoGt_a.createIndex()
        cocoGt_b.createIndex()

        cocoDt_b = cocoGt_a.loadRes(deepcopy(dict_b["annotations"]))
        cocoDt_a = cocoGt_b.loadRes(deepcopy(dict_a["annotations"]))

        cocoEval_a = COCOeval(cocoGt_a, cocoDt_b, ann_type)
        cocoEval_b = COCOeval(cocoGt_b, cocoDt_a, ann_type)

        cocoEval_a.params.iouThrs = iou_thresholds
        cocoEval_b.params.iouThrs = iou_thresholds
        cocoEval_a.params.maxDets = [100, 300, 1000]
        cocoEval_b.params.maxDets = [100, 300, 1000]

        cocoEval_a.evaluate()
        cocoEval_b.evaluate()
        cocoEval_a.accumulate()
        cocoEval_b.accumulate()

    mmap_dict = {}

    for thrs in iou_thresholds:
        mmap_dict[str(thrs)] = (compute_ap_at_iou(cocoEval_a, thrs) + compute_ap_at_iou(cocoEval_b, thrs)) / 2.0

    return mmap_dict
def compute_ap_at_iou(cocoEval, iouThr, areaRng='all', maxDets=1000):
    """
    Compute Average Precision (AP) at a specific IoU threshold for a given COCO evaluation object.

    Parameters:
        cocoEval: COCO evaluation object after `accumulate()` has been called.
        iouThr: Specific IoU threshold to compute AP for (e.g., 0.5 or 0.75).
        areaRng: Area range ('all', 'small', 'medium', 'large'). Default is 'all'.
        maxDets: Maximum number of detections. Default is 100.

    Returns:
        Average Precision (AP) value at the specified IoU threshold.
    """
    if not cocoEval.eval:
        raise Exception("The cocoEval object must have `accumulate()` run first.")

    p = cocoEval.params

    # Find the index of the specified IoU threshold
    if iouThr not in p.iouThrs:
        raise ValueError(f"iouThr={iouThr} not found in cocoEval.params.iouThrs")
    t = np.where(np.isclose(p.iouThrs, iouThr))[0][0]  # Index for the IoU threshold

    # Find the index of the specified area range
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    if not aind:
        raise ValueError(f"areaRng={areaRng} not found in cocoEval.params.areaRngLbl")
    aind = aind[0]

    # Find the index of the specified max detections
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if not mind:
        raise ValueError(f"maxDets={maxDets} not found in cocoEval.params.maxDets")
    mind = mind[0]

    # Get precision values
    s = cocoEval.eval['precision']  # Shape: [TxRxKxAxM]
    if s is None:
        raise ValueError("Precision metrics not found in cocoEval.eval")

    # Extract precision for the given IoU threshold, area range, and max detections
    s = s[t, :, :, aind, mind]
    if len(s[s > -1]) == 0:
        return -1  # No valid results

    return np.mean(s[s > -1])

def calculate_bbox_area(bbox, image_width, image_height):
    """Calculate area of bounding box from relative xywh format."""
    _, _, w, h = bbox  # Assuming bbox = (x_center, y_center, width, height)
    absolute_width = w * image_width
    absolute_height = h * image_height
    return absolute_width * absolute_height

def get_bbox_from_polygon(polygon):
    """
    Calculate the bounding box (x, y, w, h) from a polygon.

    Args:
        polygon (list): A list of points, where each point is [x, y].

    Returns:
        list: The bounding box [x_min, y_min, width, height] in relative format.
    """
    # Flatten the polygon into a list of points
    all_points = [point for shape in polygon for point in shape]

    # Extract x and y coordinates
    x_coords = all_points[::2]  # Every other element starting at 0
    y_coords = all_points[1::2]  # Every other element starting at 1

    # Get the bounding box coordinates
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    # Calculate width and height
    width = x_max - x_min
    height = y_max - y_min

    return [x_min, y_min, width, height]

def place_mask_in_image(binary_mask, bbox, image_shape):
    """
    Places a binary mask into the correct position within an image canvas based on the bounding box,
    with safety checks for alignment and boundary conditions.

    Args:
        binary_mask (np.ndarray): Binary mask (height, width) of the object.
        bbox (list or tuple): Bounding box in [x, y, width, height] format, relative to the image dimensions.
        image_shape (int, int)
            image_height (int): Height of the full image.
            image_width (int): Width of the full image.

    Returns:
        np.ndarray: Full-sized binary mask with the object mask placed in the correct position.
    """
    # Convert relative bounding box to absolute pixel values
    image_height, image_width = image_shape
    x1_f, y1_f = bbox[0] * image_width, bbox[1] * image_height
    x2_f, y2_f = (bbox[0] + bbox[2]) * image_width, (bbox[1] + bbox[3]) * image_height

    x1, x2 = int(np.floor(x1_f)), int(np.ceil(x2_f))
    y1, y2 = int(np.floor(y1_f)), int(np.ceil(y2_f))

    # Adjust if there's a mismatch in broadcasting dimensions
    if x2 - x1 != binary_mask.shape[1]:
        if abs(x1_f - x1) <= abs(x2 - x2_f):
            x1 = max(0, x1 - 1 if abs(x1_f - x1) > 1 else x1)
        else:
            x2 = min(image_width, x2 + 1 if abs(x2 - x2_f) > 1 else x2)

    if y2 - y1 != binary_mask.shape[0]:
        if abs(y1_f - y1) <= abs(y2 - y2_f):
            y1 = max(0, y1 - 1 if abs(y1_f - y1) > 1 else y1)
        else:
            y2 = min(image_height, y2 + 1 if abs(y2 - y2_f) > 1 else y2)

    # Ensure bounding box is within image boundaries
    x1, x2 = max(0, x1), min(image_width, x2)
    y1, y2 = max(0, y1), min(image_height, y2)

    # Resize the binary mask to fit the adjusted bounding box dimensions
    adjusted_mask_height = y2 - y1
    adjusted_mask_width = x2 - x1
    resized_mask = np.zeros((adjusted_mask_height, adjusted_mask_width), dtype=np.uint8)

    # Ensure the resized mask fits correctly within the adjusted region
    resized_mask[:binary_mask.shape[0], :binary_mask.shape[1]] = binary_mask[:adjusted_mask_height, :adjusted_mask_width]

    # Initialize a blank canvas for the full image
    full_image_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Place the adjusted mask in the correct location
    full_image_mask[y1:y1 + adjusted_mask_height, x1:x1 + adjusted_mask_width] = resized_mask

    return full_image_mask

class ConvergenceThresholdPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="convergence_threshold_panel",
            label="Convergence Threshold Panel",
            allow_multiple=False,
            #surfaces="modal",
            help_markdown="A panel to select and visualize the convergence threshold.",
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def render(self, ctx):
        panel = types.Object()

        v_stack = panel.v_stack(
            "v_stack", width=95, align_x="center", align_y="center"
        )
        # include plot here

        v_stack.plot(
            "violin",
            layout={
              "title": ctx.panel.state.plot_title,
              "automargin": True,
              "margin": {"t": 75}
            },
            width=90,
        )

        # define buttons here

        h_stack = v_stack.h_stack("h_stack", gap=2, align_x="center")

        h_stack.btn(
            "select_mmap",
            label="Select all mmAP",
            on_click=self.select_all_mmap,
        )
        h_stack.btn(
            "select_iaa",
            label="Select all IAA",
            on_click=self.select_all_iaa,
        )
        h_stack.btn(
            "reset",
            label="reset selection",
            on_click=self.reset_selection,
        )
        h_stack.btn(
            "apply",
            label="apply selection",
            on_click=self.apply_selection,
        )
        ctx.panel.state.compute_mean = ctx.panel.get_state("compute_mean", False)
        v_stack.bool(
            "compute_mean",
            label="Compute Means",
            on_change=self.on_change_compute_mean,
            description="Setting this option will calculate the mean for samples with the same k, n and s across thresholds."
                        "This will result in something like mmAP@[0.5,0.55,0.6, ...], similar to the standard COCO mAP metric.",
        )


        table = types.TableView()

        # Add columns for the table
        table.add_column("evaluation_type", label="Evaluation Type")
        table.add_column("annotation_type", label="Annotation Type")
        table.add_column("iou_threshold", label="IoU Threshold")
        table.add_column("mean", label="Mean")
        table.add_column("min", label="Min")
        table.add_column("max", label="Max")
        table.add_column("ci_l", label="CI L")
        table.add_column("ci_u", label="CI U")
        table.add_column("num_samples", label="Number of Samples (k)")
        table.add_column("subset_size", label="Subset Size (n)")
        table.add_column("random_seed", label="Random Seed (s)")
        table.add_column("visualized", label="Visualized")

        # Add an action column to toggle inclusion
        table.add_row_action(
            "toggle_inclusion",
            self.handle_toggle_action,
            label="Set/Unset Visualization",  # Default label for excluded
            tooltip="Click to include or exclude this row",
        )


        v_stack.obj(
            name="table",
            view=table,
            label="Convergence Threshold Table",
        )

        return types.Property(
            v_stack,
            view=types.GridView(align_x="center", align_y="center", width=100, height=100),
        )

    def on_load(self, ctx):
        # Set up data to populate the table
        dataset = ctx.dataset
        table_data = []
        visualization_states = ctx.panel.get_state("visualization_states", {})
        ctx.panel.state.plot_title = ctx.panel.get_state("plot_title", "No plot applied yet")

        if "mmAPs" in dataset.info:
            mmaps = defaultdict(list)
            for key, value in dataset.info["mmAPs"].items():
                key, _ = key.rsplit("_", 1)
                mmaps[key].append(value)
            for key, values in mmaps.items():
                ann_type, iou_threshold, random_seed, subset_n = key.split("_")
                row_key = f"mmAP_{key}"
                if row_key in visualization_states:
                    vis_state = visualization_states[row_key]
                else:
                    vis_state = "x"
                    visualization_states[row_key] = vis_state
                table_data.append({
                    "key": row_key,
                    "evaluation_type": "mmAP",
                    "annotation_type": ann_type,
                    "iou_threshold": iou_threshold,
                    "num_samples": str(len(values)),
                    "subset_size": subset_n,
                    "random_seed": random_seed,
                    "mean": round(sum(values) / len(values),3 ),
                    "min": round(min(values), 3),
                    "max": round(max(values),3 ),
                    "ci_l": round(np.percentile(values, 2.5), 3),
                    "ci_u": round(np.percentile(values, 97.5), 3),
                    "values": values,
                    "visualized": vis_state,
                })
        if "iaa_sampled" in dataset.info:
            iaas = defaultdict(list)
            for key, value in dataset.info["iaa_sampled"].items():
                key, _ = key.rsplit("_", 1)
                iaas[key].append(value)
            for key, values in iaas.items():
                ann_type, iou_threshold, random_seed, subset_n = key.split("_")
                row_key = f"IAA_{key}"
                if row_key in visualization_states:
                    vis_state = visualization_states[row_key]
                else:
                    vis_state = "x"
                    visualization_states[row_key] = vis_state
                table_data.append({
                    "key": row_key,
                    "evaluation_type": "IAA",
                    "annotation_type": ann_type,
                    "iou_threshold": iou_threshold,
                    "num_samples": str(len(values)),
                    "subset_size": subset_n,
                    "random_seed": random_seed,
                    "mean": round(sum(values) / len(values),3 ),
                    "min": round(min(values), 3),
                    "max": round(max(values),3 ),
                    "ci_l": round(np.percentile(values, 2.5), 3),
                    "ci_u": round(np.percentile(values, 97.5), 3),
                    "values": values,
                    "visualized": vis_state,
                })

        ctx.panel.state.visualization_states = visualization_states
        ctx.panel.state.table = table_data

    def handle_toggle_action(self, ctx):
        """
        Handle the toggle action to include/exclude a row.
        """
        # Get the row index and current state
        row_index = ctx.params["row"]
        table = ctx.panel.state.table
        row_key = table[row_index]["key"]

        if table[row_index]["visualized"] == "x":
            new_state = "✓"
        else:
            new_state = "x"

        table[row_index]["visualized"] = new_state
        visualization_states = ctx.panel.state.visualization_states
        visualization_states[row_key] = new_state
        ctx.panel.state.visualization_states = visualization_states

        # Toggle the state
        ctx.panel.state.table = table

    def select_all_mmap(self, ctx):
        table = ctx.panel.state.table
        visualization_states = ctx.panel.state.visualization_states
        for row in table:
            if "mmAP" == row["evaluation_type"]:
                row["visualized"] = "✓"
                visualization_states[row["key"]] = "✓"
        ctx.panel.state.table = table
        ctx.panel.state.visualization_states = visualization_states

    def select_all_iaa(self, ctx):
        table = ctx.panel.state.table
        visualization_states = ctx.panel.state.visualization_states
        for row in table:
            if "IAA" == row["evaluation_type"]:
                row["visualized"] = "✓"
                visualization_states[row["key"]] = "✓"
        ctx.panel.state.table = table
        ctx.panel.state.visualization_states = visualization_states

    def reset_selection(self, ctx):
        table = ctx.panel.state.table
        visualization_states = ctx.panel.state.visualization_states
        for row in table:
            row["visualized"] = "x"
            visualization_states[row["key"]] = "x"
        ctx.panel.state.table = table
        ctx.panel.state.visualization_states = visualization_states

    def on_change_compute_mean(self, ctx):
        current_state = ctx.params.get("value", None)

    def apply_selection(self, ctx):
        table = ctx.panel.state.table
        compute_mean = ctx.panel.state.compute_mean

        # Dictionary to group rows based on criteria
        grouped_rows = defaultdict(list)
        for row in table:
            if "✓" == row["visualized"]:
                # Create a key to group rows
                grouping_key = (
                    row["subset_size"],
                    row["random_seed"],
                    row["annotation_type"],
                    row["num_samples"],
                )
                grouped_rows[grouping_key].append(row)

        traces = []
        title_names = []

        # Process each group
        for grouping_key, rows in grouped_rows.items():
            # Ensure the group has more than one row to compute means across thresholds
            if len(rows) > 1 and compute_mean:
                # Accumulate values for each threshold
                aggregated_values = np.array([row["values"] for row in rows])
                mean_values = np.mean(aggregated_values, axis=0)
                iou_thresholds = [float(row["iou_threshold"]) for row in rows]
                sorted(iou_thresholds)

                if rows[0]["annotation_type"] == "bounding box":
                    postfix = "bb"
                else:
                    postfix = "segm"

                name = f"{rows[0]['evaluation_type']}-{postfix}@{iou_thresholds})"
                title_names.append(name)
                # Add a trace for the group
                traces.append({
                    "type": "violin",
                    "y": mean_values,
                    "name": name,
                    "box": {"visible": True},
                })
            else:
                for row in rows:
                    if row["annotation_type"] == "bounding box":
                        postfix = "bb"
                    else:
                        postfix = "segm"
                    # If only one row in the group, just add it directly
                    name = row["evaluation_type"] + "-" + postfix + "@" + str(row["iou_threshold"])
                    title_names.append(name)
                    traces.append({
                        "type": "violin",
                        "y": row["values"],
                        "name": name,
                        "box": {"visible": True},
                    })

        visualization_states = ctx.panel.state.visualization_states

        ctx.ops.clear_panel_data()
        ctx.ops.clear_panel_state()

        # Update panel state with the new traces
        ctx.panel.state.plot_title = "Convergence Threshold for at: " + str(title_names)
        ctx.panel.state.table = table
        ctx.panel.state.compute_mean = compute_mean
        ctx.panel.state.visualization_states = visualization_states
        ctx.panel.data.violin = traces

class RunErrorAnalysis(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="run_error_analysis",
            label="Run Error Analysis",
            description="Runs a Heuristic for error analysis",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            #dynamic=True,
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def __call__(self, sample_collection, annotation_type, iou_thresholds, delegate=False, recalculate=True, cache_path=None):
        ctx = dict(view=sample_collection.view())
        params = dict(annotation_type=annotation_type, iou_thresholds=iou_thresholds, delegate=delegate, api_call=True, recalculate=recalculate, cache_path=cache_path)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # Check available annotation types (bbox, polygon, mask)
        available_types = check_available_annotation_types(ctx)
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            assert ctx.params.get("annotation_type") in available_types, \
                "Annotation type {} not in {}.".format(ctx.params.get("annotation_type"), available_types)
            return None
        # --- for SDK call ---

        inputs = types.Object()

        inputs.md("###### Options for running error analysis", name="mk1")

        # Create checkboxes for available annotation types
        annotation_types_radio_group = types.RadioGroup()
        for annotation_type in available_types:
            annotation_types_radio_group.add_choice(annotation_type, label=annotation_type)

        inputs.enum(
            "annotation_type",
            annotation_types_radio_group.values(),
            label="Annotation Type",
            description="Select the annotation type to include in the analysis:",
            types=types.RadioView(),
            default=list(available_types)[0],
        )

        inputs.list(
            "iou_thresholds",
            types.Number(min=0.01, max=0.99, float=True),
            label="IoU Thresholds",
            description="Enter IoU thresholds. Values should range between 0.01 and 0.99.",
        )

        _execution_mode(ctx, inputs)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        try:
            dataset = ctx.dataset

            # get arguments
            ann_type = ctx.params.get("annotation_type")
            iou_thresholds = ctx.params.get("iou_thresholds")
            recalculate = ctx.params.get("recalculate")

            # Allow user to override the cache path, otherwise use the default
            cache_path = ctx.params.get("cache_path", None)
            if cache_path:
                base_results_dir = cache_path
            else:
                base_results_dir = os.path.join(os.path.expanduser("~"), ".fiftyone", "plugin_cache",
                                                "error_analysis_results")

            dataset_results_dir = os.path.join(base_results_dir, dataset.name)

            known_fields = set(dataset.get_field_schema().keys())  # For efficiency
            for iou in iou_thresholds:
                key = f"{ann_type}@{iou}"
                field_name = f"errors_{key.replace('.', '_').replace(' ', '_')}"
                if field_name not in known_fields:
                    dataset.add_sample_field(field_name, fo.Field)
                    known_fields.add(field_name)

            all_matches = defaultdict(list)
            samples_with_external_data = 0

            for sample in tqdm(dataset, desc=f"Calculating {ann_type} annotation errors for {dataset.name}"):

                ## 1. Updated Recalculation Check
                calc_iou_thresholds = []
                for iou_threshold in iou_thresholds:
                    key = f"{ann_type}@{iou_threshold}"
                    field_name = f"errors_{key.replace('.', '_').replace(' ', '_')}"

                    # Decide whether to calculate based on the new granular field structure
                    if recalculate or not sample.has_field(field_name) or sample[field_name] is None:
                        calc_iou_thresholds.append(iou_threshold)

                # If there's nothing new to calculate for this sample, skip it
                if not calc_iou_thresholds:
                    continue

                ## 2. Run Analysis for required thresholds
                matches = self.analyse_sample(ctx, sample, ann_type, calc_iou_thresholds)

                ## 3. Granular Saving Loop with Fallback for data above the mongo db threshold
                for key, results_list in matches.items():
                    all_matches[key].extend(results_list)
                    field_name = f"errors_{key.replace('.', '_').replace(' ', '_')}"

                    serializable_results = serialize_all_matches(results_list)
                    try:
                        dataset.set_values(field_name, {sample.id: serializable_results}, key_field="id")
                    except (DocumentTooLarge, WriteError, ValueError):
                        samples_with_external_data += 1

                        # 1. On-demand directory creation
                        os.makedirs(dataset_results_dir, exist_ok=True)

                        file_name = f"{sample.id}_{key}.json"
                        file_path = os.path.join(dataset_results_dir, file_name)

                        # 2. More informative warning message
                        print(f"\nWarning: Result '{key}' for sample {sample.id} is too large. "
                              f"Saving to external file:\n{file_path}")

                        with open(file_path, "w") as f:
                            json.dump(serializable_results, f)

                        pointer = {"external_file": file_path}
                        dataset.set_values(field_name, {sample.id: pointer}, key_field="id")

            message = f"Error Analysis Results on {ann_type}:    \n"
            error_counter = defaultdict(int)

            for iou_threshold, errors in all_matches.items():
                for error in errors:
                    error_counter[str(error.errors) + "@" + str(iou_threshold)] += 1

            for key, val in error_counter.items():
                message += str(key) + ": " + str(val) + "    \n"

            if samples_with_external_data > 0:
                message += (f"\nNote: {samples_with_external_data} result(s) were too large and stored externally in:\n"
                            f"{dataset_results_dir}")

            dataset.save()

            print(message)

            ctx.ops.open_panel("error_analysis_panel")

            return {"message": message}
        except Exception as e:
            error_details = traceback.format_exc()
            error_message = (
                f"❌ Operator failed!\n\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Details: {e}\n\n"
                f"Full Traceback:\n-----------------\n{error_details}"
            )
            print(error_message)
            return {"message": error_message}

    def resolve_output(self, ctx):
        outputs = types.Object()

        # Display the message as a notice
        outputs.view(
            "message",
            types.Notice(label=ctx.results.get("message", "")),
        )

        return types.Property(outputs)

    def analyse_sample(self, ctx, sample, ann_type, iou_thresholds):
        matches = defaultdict(list)
        merged_id_list = []
        image_shape = sample.metadata.height, sample.metadata.width
        iscrowd = lambda l: bool(l.get_attribute_value("iscrowd", False))
        iou_kwargs = dict(iscrowd=iscrowd, error_level=1,)

        if ann_type == "bounding box":
            # iou_kwargs.update() -> nothing to update
            pass
        elif ann_type == "mask":
            iou_kwargs.update(use_masks=True, tolerance=2)
        elif ann_type == "polygon":
            #iou_kwargs.update() -> nothing to update
            pass
        else:
            raise Exception(f"Annotation type {ann_type} does not exist or is not implemented.")

        raters_by_image = sample.get_field("rater_list")
        rater_combinations = combinations(raters_by_image, 2)

        for rater_a, rater_b in rater_combinations:
            if ann_type == "bounding box":
                ann_field_a = f"detections_{rater_a}"
                ann_field_b = f"detections_{rater_b}"
                element_field = "detections"
                loc_field = "bounding_box"
                InstanceClass = fol.Detection
            elif ann_type == "mask":
                ann_field_a = f"segmentations_{rater_a}"
                ann_field_b = f"segmentations_{rater_b}"
                element_field = "detections"
                loc_field = "mask"
                InstanceClass = fol.Detection
            elif ann_type == "polygon":
                ann_field_a = f"segmentations_{rater_a}"
                ann_field_b = f"segmentations_{rater_b}"
                element_field = "polylines"
                loc_field = "points"
                InstanceClass = fol.Polyline
            else:
                raise Exception(f"Annotation type {ann_type} does not exist or is not implemented..")

            annotations_a: AnnotationList = getattr(sample.get_field(ann_field_a), element_field, [])
            annotations_b: AnnotationList = getattr(sample.get_field(ann_field_b), element_field, [])

            # 1. handle case of no labels for both:
            if annotations_a == [] and annotations_b == []:
                continue

            iou_key = lambda iou_threshold, rater: ann_type + "_IoU_" + str(iou_threshold).replace(".", ",") + "_" + rater
            id_key = lambda iou_threshold, rater: ann_type + "_ID_" + str(iou_threshold).replace(".", ",") + "_" + rater
            ae_key = lambda iou_threshold, rater: ann_type + "_" + str(iou_threshold).replace(".", ",") + "_" + rater

            # 2. handle case of rater_a has no annotations
            if annotations_a == []:
                for annotation in annotations_b:
                    for iou_threshold in iou_thresholds:
                        annotation.set_field(ae_key(iou_threshold, rater_a), "mi")
                        annotation.set_field(iou_key(iou_threshold, rater_a), _NO_MATCH_IOU)
                        annotation.set_field(id_key(iou_threshold, rater_a), _NO_MATCH_ID)
                        matches[str(ann_type) + "@" + str(iou_threshold)].append(
                            AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b, errors=["mi"],
                                            iou_threshold=iou_threshold, cls_a=None, cls_b=annotation.label, iou=None,
                                            id_a=None, id_b=annotation.id, child_ids=None,
                                            loc_a=None, loc_b=annotation.get_field(loc_field))
                        )
            # 3. handle case of rater_b has no annotations
            elif annotations_b == []:
                for annotation in annotations_a:
                    for iou_threshold in iou_thresholds:
                        annotation.set_field(ae_key(iou_threshold, rater_b), "mi")
                        annotation.set_field(iou_key(iou_threshold, rater_b), _NO_MATCH_IOU)
                        annotation.set_field(id_key(iou_threshold, rater_b), _NO_MATCH_ID)
                        matches[str(ann_type) + "@" + str(iou_threshold)].append(
                            AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b, errors=["mi"],
                                            iou_threshold=iou_threshold, cls_a=annotation.label, cls_b=None, iou=None,
                                            id_a=annotation.id, id_b=None, child_ids=None,
                                            loc_a=annotation.get_field(loc_field), loc_b=None)
                        )
            # 4. handle normal case in which the matching is ran
            else:
                    # 0. Preprocess IoU and do Merging within same classes

                    for iou_threshold in iou_thresholds:
                        for annotation in annotations_a:
                            annotation.set_field(iou_key(iou_threshold, rater_b), _NO_MATCH_IOU)
                            annotation.set_field(id_key(iou_threshold, rater_b), _NO_MATCH_ID)
                        for annotation in annotations_b:
                            annotation.set_field(iou_key(iou_threshold, rater_a), _NO_MATCH_IOU)
                            annotation.set_field(id_key(iou_threshold, rater_a), _NO_MATCH_ID)

                    #
                    # I. Evaluate correct classifications and localization within the thresholds
                    #       they are considered "bad bounding boxes" -> while they are not truely bad, this is used
                    #       to declare this type of error
                    #
                    for iou_threshold in iou_thresholds:
                        # retrieve the current set of objects by categories:
                        annotation_by_category = defaultdict(lambda: defaultdict(list))
                        for annotation in annotations_a:
                            annotation_by_category[annotation.label][rater_a].append(annotation)
                        for annotation in annotations_b:
                            annotation_by_category[annotation.label][rater_b].append(annotation)
                        for objects in annotation_by_category.values():
                            detections_a_by_class = sorted(objects[rater_a], key=iscrowd)
                            detections_b_by_class = sorted(objects[rater_b], key=iscrowd)

                            # create a list of matchable object, either the case if they are crowd object or if they
                            matchable_detections_a_by_class, _ = \
                                _get_check_matchable(detections_a_by_class, id_key(iou_threshold, rater_b), iscrowd)
                            matchable_detections_b_by_class, _ = \
                                _get_check_matchable(detections_b_by_class, id_key(iou_threshold, rater_a), iscrowd)

                            # compute_ious returns a matrix where the rows in this case correspond to the detections_a
                            # and the columns to detections_b
                            ious = foui.compute_ious(matchable_detections_a_by_class, matchable_detections_b_by_class, **iou_kwargs)
                            indices_2d = _sort_iou_and_filter_by_threshold_then_return_index(ious, iou_threshold)
                            for row, col in indices_2d:
                                det_a: InstanceClass = matchable_detections_a_by_class[row]
                                det_b: InstanceClass = matchable_detections_b_by_class[col]

                                if "merged" in det_a.tags or "merged" in det_b.tags:
                                    continue

                                if (det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID or iscrowd(det_a)) or \
                                    (det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID or iscrowd(det_b)):
                                    # if this is true this means that each of the detections is either not matched or a crowed
                                    matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                        AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                        errors=["bb"],
                                                        iou_threshold=iou_threshold, cls_a=det_a.label, cls_b=det_b.label,
                                                        iou=ious[row, col], id_a=det_a.id, id_b=det_b.id, child_ids=None,
                                                        loc_a=det_a.get_field(loc_field), loc_b=det_b.get_field(loc_field))
                                    )
                                    if det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                        det_a.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                        det_a.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                        det_a.set_field(ae_key(iou_threshold, rater_b), "bb")
                                    if det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                        det_b.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                        det_b.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                        det_b.set_field(ae_key(iou_threshold, rater_a), "bb")

                            # create a list of matchable object, either the case if they are crowd object or if they
                            matchable_detections_a_by_class, matchable_merged_a_by_class = \
                                _get_check_matchable(matchable_detections_a_by_class, id_key(iou_threshold, rater_b), iscrowd)
                            matchable_detections_b_by_class, matchable_merged_b_by_class = \
                                _get_check_matchable(matchable_detections_b_by_class, id_key(iou_threshold, rater_a), iscrowd)

                            #
                            # II. Evaluate Merged/Unmerged instances between same classes
                            #
                            # 1) compute merged detections from leftover's
                            # 2) compute iou's again, but only from merged_a <-> original_b and original_a <-> merged_b
                            # 3) check if matchings are found
                            # 4) create new "merged" detection or add the merging info to the existing detection

                            new_merged_detections_a_by_class = merge_instances(matchable_detections_a_by_class, merged_id_list, image_shape, ann_type, all_iou_thresholds=iou_thresholds, other_rater=rater_b)
                            new_merged_detections_b_by_class = merge_instances(matchable_detections_b_by_class, merged_id_list, image_shape, ann_type, all_iou_thresholds=iou_thresholds, other_rater=rater_a)

                            # create a single list of matachable objects
                            # this is a design decision to again evaluate by the highest match
                            ious = _compute_ious_efficiently(matchable_detections_a_by_class, matchable_detections_b_by_class,
                                                             matchable_merged_a_by_class + new_merged_detections_a_by_class,
                                                             matchable_merged_b_by_class + new_merged_detections_b_by_class,
                                                             **iou_kwargs)

                            matchable_detections_a = matchable_detections_a_by_class + matchable_merged_a_by_class + new_merged_detections_a_by_class
                            matchable_detections_b = matchable_detections_b_by_class + matchable_merged_b_by_class + new_merged_detections_b_by_class

                            indices_2d = _sort_iou_and_filter_by_threshold_then_return_index(ious, iou_threshold)

                            # start finding matches
                            for row, col in indices_2d:
                                det_a: InstanceClass = matchable_detections_a[row]
                                det_b: InstanceClass = matchable_detections_b[col]

                                # check if both are not merged
                                if "merged" in det_a.tags and "merged" in det_b.tags:
                                    continue

                                if not "merged" in det_a.tags and not "merged" in det_b.tags:
                                    raise Exception(
                                        "There should never be the case that two unmerged instances are attempted to"
                                        "be matched during the merging issue process.")

                                # check regular conditions
                                if (det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID or iscrowd(det_a)) or \
                                        (det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID or iscrowd(det_b)):
                                    # if this is true this means that each of the detections is either not matched or a crowed
                                    matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                        AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                        errors=["mu"],
                                                        iou_threshold=iou_threshold, cls_a=det_a.label,
                                                        cls_b=det_b.label,
                                                        iou=ious[row, col], id_a=det_a.id, id_b=det_b.id,
                                                        child_ids=det_a.merge_ids if "merged" in det_a.tags else det_b.merge_ids,
                                                        loc_a=det_a.get_field(loc_field), loc_b=det_b.get_field(loc_field))
                                    )
                                    if det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                        det_a.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                        det_a.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                        det_a.set_field(ae_key(iou_threshold, rater_b), "mu")
                                    if det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                        det_b.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                        det_b.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                        det_b.set_field(ae_key(iou_threshold, rater_a), "mu")

                                    # additional to the two usual conditions, the two single instances will get updated,
                                    # since they have been matched at this point they should not be included into the matchable
                                    # items anymore
                                    if "merged" in det_a.tags:
                                        # ensure that the merged element is not added again, and only the missing data is added
                                        merge_ids = det_a.merge_ids
                                        if det_a not in annotations_a:
                                            annotations_a.append(det_a)
                                            merged_id_list.append(tuple(merge_ids))
                                        for det_a_child in matchable_detections_a:
                                            if det_a_child.get_field("id") in merge_ids and det_a_child.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                                det_a_child.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                                det_a_child.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                                det_a_child.set_field(ae_key(iou_threshold, rater_b), "mu")
                                    if "merged" in det_b.tags:
                                        merge_ids = det_b.merge_ids
                                        if det_b not in annotations_b:
                                            annotations_b.append(det_b)
                                            merged_id_list.append(tuple(merge_ids))
                                        for det_b_child in matchable_detections_b:
                                            if det_b_child.get_field("id") in merge_ids and det_b_child.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                                det_b_child.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                                det_b_child.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                                det_b_child.set_field(ae_key(iou_threshold, rater_a), "mu")

                        # <- ident back to go out of per-class loop
                        #
                        # III. Run the same as above for I. but no for mismatched classes
                        #
                        # create a list of matchable object, either the case if they are crowd object or if they didn't match yet
                        matchable_detections_a, _ = _get_check_matchable(annotations_a, id_key(iou_threshold, rater_b), iscrowd)
                        matchable_detections_b, _ = _get_check_matchable(annotations_b, id_key(iou_threshold, rater_a), iscrowd)

                        # check for significant overlaps but classes that do not match
                        # so same as for the first set but this time looking for wrong class
                        ious = foui.compute_ious(matchable_detections_a, matchable_detections_b, **iou_kwargs)
                        indices_2d = _sort_iou_and_filter_by_threshold_then_return_index(ious, iou_threshold)

                        for row, col in indices_2d:
                            det_a: InstanceClass = matchable_detections_a[row]
                            det_b: InstanceClass = matchable_detections_b[col]

                            if "merged" in det_a.tags or "merged" in det_b.tags:
                                continue

                            # check regular conditions
                            if (det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID or iscrowd(det_a)) or \
                                    (det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID or iscrowd(det_b)):
                                matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                    AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                    errors=["wc"],
                                                    iou_threshold=iou_threshold, cls_a=det_a.label,
                                                    cls_b=det_b.label,
                                                    iou=ious[row, col], id_a=det_a.id, id_b=det_b.id,
                                                    child_ids=None,
                                                    loc_a=det_a.get_field(loc_field), loc_b=det_b.get_field(loc_field))
                                )
                                if det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                    det_a.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                    det_a.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                    det_a.set_field(ae_key(iou_threshold, rater_b), "wc")
                                if det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                    det_b.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                    det_b.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                    det_b.set_field(ae_key(iou_threshold, rater_a), "wc")

                        #
                        # IV. Run the same as for II. but now allow mismatched classes
                        #
                        # allowing matching of elements of the same class
                        # Do step 1) to 4) as above, from matching the classes. But now allow the matching of
                        # merged elements from class X and original classes from class y, so inter-class matching
                        # this would be a double error wc+mu (wrong class + merging issue)

                        # create a list of matchable object, either the case if they are crowd object or if they
                        matchable_detections_a, matchable_merged_a = _get_check_matchable(matchable_detections_a,
                                                                                          id_key(iou_threshold, rater_b),
                                                                                          iscrowd)
                        matchable_detections_b, matchable_merged_b = _get_check_matchable(matchable_detections_b,
                                                                                          id_key(iou_threshold, rater_a),
                                                                                          iscrowd)

                        merged_detections_a = merge_instances(matchable_detections_a, merged_id_list, image_shape, ann_type, all_iou_thresholds=iou_thresholds, other_rater=rater_b)
                        merged_detections_b = merge_instances(matchable_detections_b, merged_id_list, image_shape, ann_type, all_iou_thresholds=iou_thresholds, other_rater=rater_a)

                        # create a single list of matachable objects
                        # this is a design decision to again evaluate by the highest match
                        ious = _compute_ious_efficiently(matchable_detections_a, matchable_detections_b,
                                                         matchable_merged_a + merged_detections_a,
                                                         matchable_merged_b + merged_detections_b,
                                                         **iou_kwargs)

                        matchable_detections_a = matchable_detections_a + matchable_merged_a + merged_detections_a
                        matchable_detections_b = matchable_detections_b + matchable_merged_b + merged_detections_b

                        indices_2d = _sort_iou_and_filter_by_threshold_then_return_index(ious, iou_threshold)

                        # start finding matches
                        for row, col in indices_2d:
                            det_a: InstanceClass = matchable_detections_a[row]
                            det_b: InstanceClass = matchable_detections_b[col]

                            # check if both are not merged
                            if "merged" in det_a.tags and "merged" in det_b.tags:
                                continue

                            if not "merged" in det_a.tags and not "merged" in det_b.tags:
                                raise Exception(
                                    "There should never be the case that two unmerged instances are attempted to"
                                    "be matched during the merging issue process.")

                            # check regular conditions
                            if (det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID or iscrowd(det_a)) or \
                                    (det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID or iscrowd(det_b)):
                                # if this is true this means that each of the detections is either not matched or a crowed
                                matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                    AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                    errors=["mu", "wc"],
                                                    iou_threshold=iou_threshold, cls_a=det_a.label,
                                                    cls_b=det_b.label,
                                                    iou=ious[row, col], id_a=det_a.id, id_b=det_b.id,
                                                    child_ids=det_a.merge_ids if "merged" in det_a.tags else det_b.merge_ids,
                                                    loc_a=det_a.get_field(loc_field), loc_b=det_b.get_field(loc_field))
                                )
                                if det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                    det_a.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                    det_a.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                    det_a.set_field(ae_key(iou_threshold, rater_b), "wc+mu")
                                if det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                    det_b.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                    det_b.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                    det_b.set_field(ae_key(iou_threshold, rater_a), "wc+mu")

                                # additional to the two usual conditions, the two single instances will get updated,
                                # since they have been matched at this point they should not be included into the matchable
                                # items anymore
                                if "merged" in det_a.tags:
                                    # ensure that the merged element is not added again, and only the missing data is added
                                    merge_ids = det_a.merge_ids
                                    if det_a not in annotations_a:
                                        annotations_a.append(det_a)
                                        merged_id_list.append(tuple(merge_ids))
                                    for det_a_child in matchable_detections_a:
                                        if det_a_child.id in merge_ids and det_a_child.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                            det_a_child.set_field(id_key(iou_threshold, rater_b), det_b.id)
                                            det_a_child.set_field(iou_key(iou_threshold, rater_b), ious[row, col])
                                            det_a_child.set_field(ae_key(iou_threshold, rater_b), "wc+mu")
                                if "merged" in det_b.tags:
                                    # ensure that the merged element is not added again, and only the missing data is added
                                    merge_ids = det_b.merge_ids
                                    if det_b not in annotations_b:
                                        annotations_b.append(det_b)
                                        merged_id_list.append(tuple(merge_ids))
                                    for det_b_child in matchable_detections_b:
                                        if det_b_child.id in merge_ids and det_b_child.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                            det_b_child.set_field(id_key(iou_threshold, rater_a), det_a.id)
                                            det_b_child.set_field(iou_key(iou_threshold, rater_a), ious[row, col])
                                            det_b_child.set_field(ae_key(iou_threshold, rater_a), "wc+mu")

                        # rest is overlooked or unnecessary instance
                        matchable_detections_a, _ = _get_check_matchable(annotations_a, id_key(iou_threshold, rater_b), iscrowd)
                        matchable_detections_b, _ = _get_check_matchable(annotations_b, id_key(iou_threshold, rater_a), iscrowd)

                        for det_a in matchable_detections_a:
                            # don't count merged instances as mi
                            if "merged" in det_a.tags:
                                continue
                            if det_a.get_field(id_key(iou_threshold, rater_b)) == _NO_MATCH_ID:
                                matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                    AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                    errors=["mi"],
                                                    iou_threshold=iou_threshold, cls_a=det_a.label, cls_b=None,
                                                    iou=None,
                                                    id_a=det_a.id, id_b=None, child_ids=None,
                                                    loc_a=det_a.get_field(loc_field), loc_b=None)
                                )
                                det_a.set_field(ae_key(iou_threshold, rater_b), "mi")
                        for det_b in matchable_detections_b:
                            # don't count merged instances as mi
                            if "merged" in det_b.tags:
                                continue
                            if det_b.get_field(id_key(iou_threshold, rater_a)) == _NO_MATCH_ID:
                                matches[str(ann_type) + "@" + str(iou_threshold)].append(
                                    AnnotationError(sample_id=sample.id, rater_a=rater_a, rater_b=rater_b,
                                                    errors=["mi"],
                                                    iou_threshold=iou_threshold, cls_a=None, cls_b=det_b.label,
                                                    iou=None,
                                                    id_a=None, id_b=det_b.id, child_ids=None,
                                                    loc_a=None, loc_b=det_b.get_field(loc_field))
                                )
                                det_b.set_field(ae_key(iou_threshold, rater_a), "mi")

        sample.save()
        return matches


import os
import json
from collections import defaultdict


def load_error_analysis_results(dataset):
    """
    Discovers and loads all error analysis results from a dataset.

    This function automatically finds all fields matching the 'errors_*'
    pattern and correctly handles results stored in the database or in
    external JSON files.

    Args:
        dataset: a fiftyone.core.dataset.Dataset

    Returns:
        a defaultdict containing the aggregated error analysis results.
    """
    print("Discovering and loading all annotation error results...")
    all_results = defaultdict(list)
    schema = dataset.get_field_schema()

    # 1. Discover all error fields and map them back to their original keys
    error_fields_to_keys = {}
    for field_name in schema:
        if field_name.startswith("errors_"):
            # Reconstruct: "errors_bounding_box_0_5" -> "bounding box@0.5"
            key_part = field_name[len("errors_"):]
            last_underscore_index = key_part.rfind('_')

            if last_underscore_index == -1:
                continue  # Skip malformed field names

            sanitized_ann_type = key_part[:last_underscore_index]
            sanitized_iou = key_part[last_underscore_index + 1:]

            ann_type = sanitized_ann_type.replace('_', ' ')
            iou = sanitized_iou.replace('_', '.')
            original_key = f"{ann_type}@{iou}"
            error_fields_to_keys[field_name] = original_key

    if not error_fields_to_keys:
        print("No annotation error fields found in dataset.")
        return all_results

    # 2. Iterate through samples once, loading data from all discovered fields
    for sample in tqdm(dataset.view(), desc="Aggregating results"):
        for field_name, key in error_fields_to_keys.items():
            if sample.has_field(field_name):
                data = sample[field_name]

                if data is None:
                    continue

                if isinstance(data, dict) and "external_file" in data:
                    file_path = data["external_file"]
                    try:
                        with open(file_path, 'r') as f:
                            loaded_data = json.load(f)
                            all_results[key].extend(loaded_data)
                    except FileNotFoundError:
                        print(f"Warning: Could not find external file: {file_path}")
                else:
                    all_results[key].extend(data)

    return all_results

class AnnotationError():
    def __init__(self, sample_id: str, rater_a: str, rater_b: str, errors: list, iou_threshold, cls_a=None, cls_b=None,
                 iou=None, id_a=None, id_b=None, child_ids=None, loc_a: list = None, loc_b: list = None):
        self.sample_id = sample_id
        self.rater_a = rater_a
        self.rater_b = rater_b
        self.errors = errors
        self.iou_threshold = iou_threshold
        self.cls_a = cls_a
        self.cls_b = cls_b
        self.iou = iou
        self.id_a = id_a
        self.id_b = id_b
        self.child_ids = child_ids
        self.loc_a = loc_a
        self.loc_b = loc_b

    def to_dict(self):
        return {
            'sample_id': self.sample_id,
            'rater_a': self.rater_a,
            'rater_b': self.rater_b,
            'errors': self.errors,
            'iou_threshold': self.iou_threshold,
            'cls_a': self.cls_a,
            'cls_b': self.cls_b,
            'iou': self.iou,
            'id_a': self.id_a,
            'id_b': self.id_b,
            'child_ids': self.child_ids,
            'loc_a': self.loc_a,
            'loc_b': self.loc_b
        }

    @classmethod
    def from_dict(cls, data):
        # Convert 'iou' and 'iou_threshold' to floats if they are present and not None
        if 'iou' in data and data['iou'] is not None:
            data['iou'] = float(data['iou'])
        if 'iou_threshold' in data and data['iou_threshold'] is not None:
            data['iou_threshold'] = float(data['iou_threshold'])
        # Create an instance from a dictionary
        return cls(**data)

    def __repr__(self):
        return str(self.sample_id) + ":" + str(self.errors) + "-" + self.rater_a + "/" + self.rater_b

# JSON serialization
def serialize_all_matches(all_matches):
    # Convert the nested structure into a JSON-serializable format
    def convert(value):
        if isinstance(value, list):
            # Convert each item in the list
            return [item.to_dict() if isinstance(item, AnnotationError) else item for item in value]
        elif isinstance(value, dict):
            # Recursively convert each value in the dictionary
            return {key: convert(val) for key, val in value.items()}
        else:
            # Return value as-is for primitive types
            return value

    #return json.dumps(convert(all_matches))
    return convert(all_matches)

# JSON deserialization
def deserialize_all_matches(json_string):
    # Parse the JSON into a nested dictionary
    def convert(value):
        if isinstance(value, list):
            # Convert each item in the list back to AnnotationError if possible
            return [AnnotationError.from_dict(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            # Recursively convert each value in the dictionary
            return {key: convert(val) for key, val in value.items()}
        else:
            # Return value as-is for primitive types
            return value

    data = json.loads(json_string)
    return convert(data)

def _sort_iou_and_filter_by_threshold_then_return_index(ious, threshold):
    """
        Uses the IoU Matrix and a threshold to create a list of tuples. Each tuple contains (row, column)-index of the
        next highest element in the matching. The threshold ensures that only elements equal or above the iou-threshold
        are considered.
    """
    flattened_array = ious.flatten()
    sorted_indices = np.argsort(-flattened_array)
    indices_2d = [np.unravel_index(idx, ious.shape) for idx in sorted_indices if flattened_array[idx] >= threshold]
    return indices_2d

def _get_check_matchable(detections, id_key, iscrowd):
    _matchable_detections = []
    _matchable_merged_detections  = []
    try:
        for detection in detections:
            if iscrowd(detection) or detection.get_field(id_key) == _NO_MATCH_ID:
                if "merged" in detection.tags:
                    _matchable_merged_detections.append(detection)
                else:
                    _matchable_detections.append(detection)
    except Exception as e:
        error_details = traceback.format_exc()
        raise Exception(
            f"Found error with instance {detection}, not containing requested id_key {id_key}, throwing the error"
            f"{error_details}"
        )
    return _matchable_detections, _matchable_merged_detections

def _compute_ious_efficiently(matchable_detections_a, matchable_detections_b,
                               merged_detections_a, merged_detections_b, **iou_kwargs):
    ious_a_merged_to_b = foui.compute_ious(merged_detections_a, matchable_detections_b, **iou_kwargs)
    ious_b_merged_to_a = foui.compute_ious(matchable_detections_a, merged_detections_b, **iou_kwargs)

    rows = len(matchable_detections_a) +len(merged_detections_a)
    cols = len(matchable_detections_b) +len(merged_detections_b)
    ious = np.zeros((rows,cols))

    if ious.size == 0:
        return ious

    # assign to top right
    ious[0:len(matchable_detections_a), len(matchable_detections_b):] = ious_b_merged_to_a
    # assign to bottom left
    ious[len(matchable_detections_a):, 0:len(matchable_detections_b)] = ious_a_merged_to_b

    return ious

def merge_instances(detections, merged_id_list: list, image_shape, ann_type, all_iou_thresholds, other_rater):
    merged_detections = []
    combinations_to_merge = list(combinations(detections, 2))
    for det_a, det_b in combinations_to_merge:
        # dont merge instances again that already have been merged once before.
        if "merged" in det_a.tags or "merged" in det_b.tags:
            continue
        # check if this combination was already merged once:
        if (det_a.id, det_b.id) in merged_id_list:
            continue
        if det_a.label == det_b.label:
            new_instance: fol.Label = None
            attributes = {
                "tags": ["merged"],
                "label": det_a.label,
                "iscrowd": det_a.iscrowd or det_b.iscrowd,
                "rater_id": det_a.rater_id,
                "merge_ids": (det_a.id, det_b.id),
                # no annotation id
            }
            # merge
            if ann_type == "bounding box":
                # merge bounding boxes
                attributes.update(bounding_box=merge_relative_boxes(det_a, det_b))
                new_instance = fol.Detection(**attributes,)
            elif ann_type == "mask":
                # merge masks
                poly_1 = det_a.to_polyline()
                points = merge_polygons(poly_1, det_b.to_polyline())
                attributes.update(points=points, filled=poly_1.filled, closed=poly_1.closed)
                new_polygon = fol.Polyline(**attributes)
                new_instance = new_polygon.to_detection(frame_size=image_shape)
            elif ann_type == "polygon":
                points = merge_polygons(det_a, det_b)
                attributes.update(points=points, filled=det_a.filled, closed=det_a.closed)
                new_instance = fol.Polyline(**attributes)
            else:
                raise Exception(f"Annotation type {ann_type} not implemented.")

            # Add new Instance matchings for all thresholds
            iou_key = lambda iou, rater: f"{ann_type}_IoU_{str(iou).replace('.', ',')}_{rater}"
            id_key = lambda iou, rater: f"{ann_type}_ID_{str(iou).replace('.', ',')}_{rater}"
            for iou in all_iou_thresholds:
                new_instance.set_field(iou_key(iou, other_rater), _NO_MATCH_IOU)
                new_instance.set_field(id_key(iou, other_rater), _NO_MATCH_ID)

            merged_detections.append(new_instance)

    return merged_detections

def merge_relative_boxes(det_a, det_b):
    box1 = det_a.bounding_box
    box2 = det_b.bounding_box

    # Convert to absolute coordinates
    abs_box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    abs_box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Find the enclosing box
    enclosing_box = [
        min(abs_box1[0], abs_box2[0]),  # x_min
        min(abs_box1[1], abs_box2[1]),  # y_min
        max(abs_box1[2], abs_box2[2]),  # x_max
        max(abs_box1[3], abs_box2[3])  # y_max
    ]

    # Convert back to relative format
    merged_box = [
        enclosing_box[0],  # x
        enclosing_box[1],  # y
        enclosing_box[2] - enclosing_box[0],  # width
        enclosing_box[3] - enclosing_box[1]  # height
    ]

    return merged_box

def merge_polygons(det_a, det_b):
    shape_a = validate_polygon(det_a.to_shapely())
    shape_b = validate_polygon(det_b.to_shapely())

    union_shape = shape_a.union(shape_b)

    points = shapely_to_fiftyone_points(union_shape)

    return points

def validate_polygon(geom):
    """
    Validates and attempts to fix an invalid Shapely geometry.
    Applies multiple strategies to repair invalid polygons.

    Parameters:
        geom: A Shapely geometry object (Polygon or MultiPolygon).

    Returns:
        A valid Shapely geometry object or a simplified version if repair fails.
    """
    try:
        with suppress_output():
            if not geom.is_valid:
                geom = geom.buffer(0)  # Attempt to fix small invalidities
            if not geom.is_valid:
                geom = make_valid(geom)  # Attempt to fix further issues
            if not geom.is_valid:
                geom = geom.simplify(0.001, preserve_topology=True)  # Simplify if still invalid
            return geom
    except Exception as e:
        print(f"Executed non-topology-preserving simplification: {e}.")
        # Non-topology-preserving simplification as last resort
        return geom.simplify(0.001, preserve_topology=False)


def shapely_to_fiftyone_points(geometry):
    """
    Converts a Shapely Polygon or MultiPolygon to FiftyOne points format
    by removing holes and simplifying the geometry.

    Parameters:
        geometry (Polygon or MultiPolygon): The Shapely geometry to convert.

    Returns:
        list: A list of points in FiftyOne format, adhering to FiftyOne's validation.
    """
    def convert_polygon(polygon):
        # Use only the exterior ring (ignoring holes)
        return [list(coord) for coord in polygon.exterior.coords]

    if isinstance(geometry, Polygon):
        # Handle a single Polygon
        return [convert_polygon(geometry)]  # Single list of points wrapped in another list
    elif isinstance(geometry, MultiPolygon):
        # Handle a MultiPolygon by converting each sub-polygon
        return [convert_polygon(polygon) for polygon in geometry.geoms]
    else:
        raise ValueError("Input geometry must be a Shapely Polygon or MultiPolygon.")

class ErrorAnalysisPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="error_analysis_panel",
            label="Annotation Error Analysis",
            allow_multiple=False,
            surfaces="grid",
            help_markdown="A panel to visualize and select annotation errors",
            icon="/assets/icon.svg",
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def on_load(self, ctx):
        name_mapping = {
            "bb": "Correct Instance",
            "mi": "Missing/Overlooked Instance",
            "wc": "Mismatched Class Assignment",
            "mu": "Merged/Unmerged Instance"
        }
        color_mapping = {
            "bb": "#4E96BE",
            "mi": "#AF8742",
            "wc": "#56C0A3",
            "mu": "#B77542"
        }
        dataset = ctx.dataset
        annotation_errors = load_error_analysis_results(dataset)

        table_data = []
        traces = []
        for key, values in annotation_errors.items():
            ann_type, iou_threshold = key.split("@")
            ae_counter = defaultdict(int)
            total_errors = 0
            for value in values:
                errors = value["errors"]
                for error in errors:
                    ae_counter[error] += 1
                    total_errors +=1
            error_percentage = lambda err_key: str(round(ae_counter[err_key] / total_errors * 100, 2)) + "%"
            table_data.append({
                "key": key,
                "iou_threshold": ann_type,
                "annotation_type": iou_threshold,
                "bb": ae_counter["bb"],
                "bb-rel": error_percentage("bb"),
                "mi": ae_counter["mi"],
                "mi-rel": error_percentage("mi"),
                "wc": ae_counter["wc"],
                "wc-rel": error_percentage("wc"),
                "mu": ae_counter["mu"],
                "mu-rel": error_percentage("mu"),
                "total": total_errors,
            })
            for error_type, count in ae_counter.items():
                traces.append({
                    "type": "histogram",
                    "name":  name_mapping[error_type],
                    "bingroup": key,
                    "x": [key] * count,  # Ensure the errors are grouped by key
                    "marker": {"color": color_mapping[error_type]},  # Add color based on error type
                })
        ctx.panel.state.table = table_data
        ctx.panel.data.histogram = traces

        ctx.ops.split_panel("error_analysis_panel", layout="horizontal")

    def on_histogram_click(self, ctx):
        ctx.ops.notify(str(ctx.params))
        name_mapping = {
            "Correct Instance": "bb",
            "Missing/Overlooked Instance": "mi",
            "Mismatched Class Assignment": "wc",
            "Merged/Unmerged Instance": "mu"
        }
        key = ctx.params.get("x")
        error_type = name_mapping[ctx.params.get("trace")]
        ann_type, iou_threshold = key.split("@")

        field_set = set()
        rater_set = set()
        for sample in ctx.dataset:
            for rater in sample.rater_list:
                if ann_type == "bounding box":
                    ann_field = f"detections_{rater}"
                elif ann_type == "mask":
                    ann_field = f"segmentations_{rater}"
                elif ann_type == "polygon":
                    ann_field = f"segmentations_{rater}"
                else:
                    raise Exception(f"Annotation type {ann_type} does not exist or is not implemented..")
                field_set.add(ann_field)
                rater_set.add(rater)

        view = ctx.dataset.select_fields(list(field_set))

        for field in field_set:
            for rater in rater_set:
                if rater in field:
                    continue
                filter = ann_type + "_" + str(iou_threshold).replace(".", ",") + "_" + rater
                view = view.filter_labels(
                    field,
                    fo.ViewField(filter) == error_type
                )

        if view is not None:
            ctx.ops.set_view(view=view)

    def render(self, ctx):
        panel = types.Object()

        panel.plot(
            "histogram",
            layout={
                "title": "Annotation Error Distribution",
                "automargin": True,
                "barmode": "stack",
            },
            width=95,
            on_click=self.on_histogram_click
        )

        table = types.TableView()

        table.add_column("annotation_type", label="Annotation Type")
        table.add_column("iou_threshold", label="IoU Threshold")
        table.add_column("bb", label="Correct Instances")
        table.add_column("bb-rel", label="C.I. %")
        table.add_column("mi", label="Missing/ Overlooked I.")
        table.add_column("mi-rel", label="M/O %")
        table.add_column("wc", label="Mismatched Class A.")
        table.add_column("wc-rel", label="M.C. %")
        table.add_column("mu", label="Merged/ Unmerged I.")
        table.add_column("mu-rel", label="M/U %")

        panel.obj(
            name="table",
            view=table,
            label="Annotation Errors"
        )

        return types.Property(
            panel,
            view=types.GridView(align_x="center", align_y="center", width=100, height=100),
        )

def check_available_annotation_types(ctx):
    available_types = ctx.params.get("available_types", [])
    if available_types == []:
        dataset = ctx.dataset
        available_types = dataset.info["ann_types"]
        ctx.params["available_types"] = available_types
    return available_types

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )

@contextmanager
def suppress_output():
    """
    Suppresses all output (stdout and stderr) in both Jupyter Notebook and standard Python.
    """
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

def debug_helper(ctx, msg):
    print(str(msg))
    ctx.ops.notify(str(msg))

def _sanitize_for_field_name(name: str) -> str:
    """Replaces characters that are invalid for FiftyOne field names."""
    if not isinstance(name, str):
        name = str(name)
    return name.replace('.', '_')

def register(plugin):
    plugin.register(LoadMultiAnnotatedData)
    plugin.register(CalculateIaa)
    plugin.register(IAAPanel)
    plugin.register(CalculatemmAP)
    plugin.register(ConvergenceThresholdPanel)
    plugin.register(RunErrorAnalysis)
    plugin.register(ErrorAnalysisPanel)