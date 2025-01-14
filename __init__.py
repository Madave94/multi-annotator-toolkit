import fiftyone as fo
import fiftyone.operators as foo
import numpy as np
from fiftyone.operators import types
from fiftyone import ViewField as F
from collections import defaultdict
import os
import json
from tqdm import tqdm
import random
import traceback
from pycocotools import mask as coco_mask
import numpy as np
from copy import deepcopy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from kalphacv import reliability_data, krippendorff_alpha

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
        )
        ```

        **2. Use the Operator via the SDK:**

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
            rater_list = file_name_to_rater_list.get(sample_file_name)

            if rater_list is not None:
                # Add the rater_list to the mapping
                id_to_rater_list[sample.id] = rater_list
            else:
                # Handle the case where there's no matching rater_list
                print(f"No rater_list found for {sample_file_name}")

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
        messages = [
                f"Successfully loaded multi-annotations for {num_updated} out of {len(dataset)} samples.\n"
            ]

        if 'detections' in field_schema:
            detection_counts = split_annotations_by_rater(dataset, 'detections')
            messages.append(
                f"Detections - Total: {detection_counts['total_annotations']}, "
                f"Moved: {detection_counts['annotations_moved']}, "
                f"Unassigned: {detection_counts['annotations_unassigned']}."
            )
        else:
            print("No 'detections' field found in the dataset.")

        if 'segmentations' in field_schema:
            segmentation_counts = split_annotations_by_rater(dataset, 'segmentations')
            messages.append(
                f"Segmentations - Total: {segmentation_counts['total_annotations']}, "
                f"Moved: {segmentation_counts['annotations_moved']}, "
                f"Unassigned: {segmentation_counts['annotations_unassigned']}."
            )
        else:
            print("No 'segmentations' field found in the dataset.")

        # **Join the messages into a single string**
        message_str = "\n".join(messages)

        print(message_str)

        return {
            "message": message_str,
            "num_updated": num_updated,
            "num_samples": len(dataset),
        }

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
def split_annotations_by_rater(dataset, source_field, field_prefix=None):
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

    # Collect field type (fo.Detections or fo.Polylines, etc.)
    field_schema = dataset.get_field_schema()

    # Process each sample
    for sample in tqdm(dataset, desc=f"Processing {source_field}"):
        rater_list = sample.get_field("rater_list")
        annotations = sample.get_field(source_field)
        if annotations is None:
            continue

        # Determine the attribute to access based on field type
        if isinstance(annotations, fo.Detections):
            annotations_list = annotations.detections
        elif isinstance(annotations, fo.Polylines):
            annotations_list = annotations.polylines
        else:
            raise Exception("Invalid annotations type processed. Should be detections or polylines.")

        # Initialize per-rater annotations dict
        annotations_by_rater = {rater_id: [] for rater_id in rater_list}
        unassigned_annotations = []

        # Process annotations
        for annotation in annotations_list:
            total_annotations += 1
            rater_id = annotation.get_field('rater_id')
            if rater_id and rater_id in rater_list:
                annotations_by_rater[rater_id].append(annotation)
                annotations_moved += 1
            else:
                unassigned_annotations.append(annotation)
                annotations_unassigned += 1


        # Assign per-rater annotations to new fields
        for rater_id, ann_list in annotations_by_rater.items():
            field_name = f"{field_prefix}{rater_id}"
            # Ensure field exists in dataset schema
            if field_name not in field_schema:
                dataset.add_sample_field(
                    field_name,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=type(annotations)
                )
            if ann_list:
                if isinstance(annotations, fo.Detections):
                    sample[field_name] = fo.Detections(detections=ann_list)
                elif isinstance(annotations, fo.Polylines):
                    sample[field_name] = fo.Polylines(polylines=ann_list)
                else:
                    raise Exception("Invalid annotations type processed. Should be detections or polylines.")
            else:
                sample[field_name] = None  # Clear field if empty

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

class CalculateIaa(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="calculate_iaa",
            label="Calculate IAA",
            description="Calculates the Inter-Annotator-Agreement",
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
            dynamic=True,
        )

    def __call__(self, sample_collection, annotation_type, iou_thresholds, run_sampling=False, subset_n=None,
                 sampling_k=None, random_seed_s=None, delegate=False):
        ctx = dict(view=sample_collection.view())
        params = dict(annotation_type=annotation_type, iou_thresholds=iou_thresholds, run_sampling=run_sampling,
                      subset_n=subset_n, sampling_k=sampling_k, random_seed_s=random_seed_s, delegate=delegate, api_call=True)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            return None
        # --- for SDK call ---

        inputs = types.Object()

        inputs.md("###### Options for calculating inter annotator agreement", name="mk1")

        # Check available annotation types (bbox, polygon, mask)
        available_types = check_available_annotation_types(ctx)

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
            subset_n = ctx.params.get("subset_n")
            sampling_k = ctx.params.get("sampling_k")
            random_seed_s = ctx.params.get("random_seed_s")
            if "iaa_sampled" not in dataset.info:
                dataset.info["iaa_sampled"] = {}
                dataset.save()
            iaas = dataset.info["iaa_sampled"]
            random.seed(random_seed_s)
            for idx in range(sampling_k):
                # sample iaa value per threshold
                indices = random.sample(range(len(dataset)), subset_n)
                for iou_threshold in iou_thresholds:
                    iaa_values = [alphas[str(iou_threshold)][i] for i in indices]
                    iaas[f"{ann_type}_{iou_threshold}_{random_seed_s}_{subset_n}_{idx}"] = sum(iaa_values) / len(iaa_values)

            dataset.info["iaa_sampled"] = iaas
            dataset.save()

        message = "Mean K-Alpha for:"
        for iou_threshold in iou_thresholds:
            u_k_alpha = sum(alphas[str(iou_threshold)]) / len(alphas[str(iou_threshold)])
            message += f"\n\tIoU {iou_threshold} on {ann_type}: {u_k_alpha}"

        # Include a message for sampling
        if run_sampling:
            message += f"\nSampling completed with {sampling_k} samples of size {subset_n}"

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
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def on_load(self, ctx):
        iaa_list = ctx.dataset.info["iaa_analyzed"]

        iaa_dict = defaultdict(list)
        for iaa in iaa_list:
            ann_type, iou = iaa.split("-")
            iaa_dict[ann_type].append(iou)

        ctx.panel.state.iaa_dict = iaa_dict
        if ctx.panel.state.ann_type_selection is None:
            ctx.panel.state.ann_type_selection = list(iaa_dict.keys())[0]
        if ctx.panel.state.iou_selection is None:
            ctx.panel.state.iou_selection = iaa_dict[ctx.panel.state.ann_type_selection][0]

        values = self.get_values(ctx)
        ctx.panel.state.plot_title = "Inter-Annotat-Agreement: {} {}".format(
            ctx.panel.state.ann_type_selection,
            ctx.panel.state.iou_selection)
        ctx.panel.data.histogram = {"x": values,
                                    "type": "histogram",
                                    "marker": {"color": "#FF6D05"}, # gray #808080
                                    "xbins": {"end": 1.0, "size": 0.1},
                                    }

        ctx.panel.state.mean_msg = "Mean for {} annotations wih iou-threshold {}: **{:.3f}**".format(ctx.panel.state.ann_type_selection,
                                                                                                 ctx.panel.state.iou_selection,
                                                                                                 (sum(values) / len(values)))

        ctx.ops.split_panel("iaa_panel", layout="horizontal")

    def change_ann_type(self, ctx):
        ctx.panel.state.ann_type_selection = ctx.params["value"]

    def change_iou_value(self, ctx):
        ctx.panel.state.iou_selection = ctx.params["value"]

    def get_values(self, ctx):
        values = []
        for sample in ctx.dataset:
            values.append(
                sample["iaa"][ctx.panel.state.ann_type_selection + "-" + ctx.panel.state.iou_selection]
            )
        return values

    def on_histogram_click(self, ctx):
        bin_range = ctx.params.get("range")
        min_value = bin_range[0]
        max_value = bin_range[1]

        ann_type = ctx.panel.state.ann_type_selection
        iou_value = ctx.panel.state.iou_selection
        field_name = "iaa.{}-{}".format(ann_type, iou_value)

        view = ctx.dataset.match((F(field_name) >= min_value) & (F(field_name) <= max_value))

        if view is not None:
            ctx.ops.set_view(view=view)

    def render(self, ctx):
        panel = types.Object()

        h_stack = panel.h_stack("h_stack", align_x="center", align_y="center", gap=5)

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
            label="load with set values",
            on_click=self.on_load
        )

        panel.plot(
            "histogram",
            layout={
                "title": {
                    "text": ctx.panel.state.plot_title,
                    "automargin": True,
                },
                "xaxis": {"title": "K-Alpha", },
                "yaxis": {"title": "Count"},
                "bargap": 0.05,
            },
            on_click=self.on_histogram_click,

        )

        v_stack = panel.v_stack("v_stack", align_x="center", align_y="center", width=75)
        v_stack.md(ctx.panel.state.mean_msg)

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
            light_icon="/assets/icon-light.svg",
            dark_icon="/assets/icon-dark.svg",
        )

    def __call__(self, sample_collection, annotation_type, iou_thresholds, dataset_scope, subset_n=None, sampling_k=None,
                 random_seed_s=None, delegate=False):
        ctx = dict(view=sample_collection.view())
        params = dict(annotation_type=annotation_type, iou_thresholds=iou_thresholds, dataset_scope=dataset_scope,
                      subset_n=subset_n, sampling_k=sampling_k, random_seed_s=random_seed_s, delegate=delegate, api_call=True)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # --- for SDK call ---
        api_call = ctx.params.get("api_call", False)
        if api_call:
            # Parameters are already provided; no need to resolve input
            return None
        # --- for SDK call ---

        inputs = types.Object()

        inputs.md("###### Options for calculating modified mean Average Precision", name="mk1")

        dataset = ctx.dataset

        # Check available annotation types (bbox, polygon, mask)
        available_types = check_available_annotation_types(ctx)

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
        for image_idx, sample in enumerate(dataset):
            raters_by_image = sample.get_field("rater_list")
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
                        mask = place_mask_in_image(annotation["mask"], bbox, height, width)
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
            raise Exception("this should not happen")
        # check the sampling
        if dataset_scope_choice == "Full":
            ann_dict_0 = {"annotations": annotations_dict["0"], "images": annotations_dict["images"], "categories": categories}
            ann_dict_1 = {"annotations": annotations_dict["1"], "images": annotations_dict["images"], "categories": categories}
            mmap_dict = calc_mmap(ann_dict_0, ann_dict_1, annType, iou_thresholds, ctx)
            for thrs in iou_thresholds:
                mmaps[f"{ann_type}_{thrs}"] = mmap_dict[str(thrs)]
        elif dataset_scope_choice == "Partial":
            random.seed(random_seed_s)
            for idx in range(sampling_k):
                sampled_images = random.sample(annotations_dict["images"], subset_n)
                image_ids = [image["id"] for image in sampled_images]
                ann_0 = [annotation for annotation in annotations_dict["0"] if annotation["image_id"] in image_ids]
                ann_1 = [annotation for annotation in annotations_dict["1"] if annotation["image_id"] in image_ids]
                ann_dict_0 = {"annotations": ann_0, "images": sampled_images, "categories": categories}
                ann_dict_1 = {"annotations": ann_1, "images": sampled_images, "categories": categories}
                mmap_dict = calc_mmap(ann_dict_0, ann_dict_1, annType, iou_thresholds, ctx)
                for thrs in iou_thresholds:
                    mmaps[f"{ann_type}_{thrs}_{random_seed_s}_{subset_n}_{idx}"] = mmap_dict[str(thrs)]
        else:
            raise Exception("This should not be possible.")

        dataset.info["mmAPs"] = mmaps
        dataset.save()

        return {}

    def resolve_output(self, ctx):
        outputs = types.Object()

        ### Add your outputs here ###

        return types.Property(outputs)

def calc_mmap(dict_a, dict_b, ann_type, iou_thresholds, ctx):
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

def place_mask_in_image(binary_mask, bbox, image_height, image_width):
    """
    Places a binary mask into the correct position within an image canvas based on the bounding box,
    with safety checks for alignment and boundary conditions.

    Args:
        binary_mask (np.ndarray): Binary mask (height, width) of the object.
        bbox (list or tuple): Bounding box in [x, y, width, height] format, relative to the image dimensions.
        image_height (int): Height of the full image.
        image_width (int): Width of the full image.

    Returns:
        np.ndarray: Full-sized binary mask with the object mask placed in the correct position.
    """
    # Convert relative bounding box to absolute pixel values
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

def check_available_annotation_types(ctx):
    available_types = ctx.params.get("available_types", [])
    dataset = ctx.dataset
    if available_types == []:
        sample = dataset.first()
        # Check if rater list exists
        if not sample.has_field("rater_list"):
            inputs = types.Object()
            prop = inputs.view("message", types.Error(
                label="No multi-annotated data found."),
                               description="Please run `Load Multi Annotated Data` first or check if your annotations file is properly"
                                           " formatted.",
                               )
            prop.invalid = True
            return types.Property(inputs)

        raters_by_image = sample.get_field("rater_list")

        # Check for bounding boxes
        for rater_id in raters_by_image:
            detections = sample.get_field(f"detections_{rater_id}")
            if detections is not None:
                available_types.append("bounding box")
                break

        # Check for segmentations (masks and polygons)
        found_segmentation = False
        for rater_id in raters_by_image:
            segmentations = sample.get_field(f"segmentations_{rater_id}")
            if segmentations is not None:
                if hasattr(segmentations, "detections"):
                    for detection in segmentations.detections:
                        if "bounding_box" in detection:
                            available_types.append("mask")
                            found_segmentation = True
                            break
                    if found_segmentation:
                        break
                if hasattr(segmentations, "polylines"):
                    for polyline in segmentations.polylines:
                        available_types.append("polygon")
                        found_segmentation = True
                        break
                    if found_segmentation:
                        break
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

def register(plugin):
    plugin.register(LoadMultiAnnotatedData)
    plugin.register(CalculateIaa)
    plugin.register(IAAPanel)
    plugin.register(CalculatemmAP)