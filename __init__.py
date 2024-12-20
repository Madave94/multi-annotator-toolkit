import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F
from collections import defaultdict
import os
import json
from tqdm import tqdm
import traceback
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
        )

    def __call__(self, sample_collection, annos_path, overwrite=False, num_workers=False, delegate=False):
        ctx = dict(view=sample_collection.view())
        params = dict(annos_path=annos_path, overwrite=overwrite, num_workers=num_workers, delegate=delegate)
        return foo.execute_operator(self.uri, ctx, params=params)

    def resolve_input(self, ctx):
        # --- for SDK call ---
        annos_path = ctx.params.get("annos_path", None)
        if annos_path is not None:
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
    if dataset.match(F(source_field).exists()).count() == 0:
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
        )

    def __call__(self, sample_collection, sample_selection, annotation_type, iou_thresholds, delegate=False):
        ctx = dict(view=sample_collection.view())
        params = dict(sample_selection=sample_selection, annotation_type=annotation_type, iou_thresholds=iou_thresholds, delegate=delegate)
        return foo.execute_operator(self.uri, ctx, params=params)



    def resolve_input(self, ctx):
        # --- for SDK call ---
        iou_thresholds = ctx.params.get("iou_thresholds", None)
        if iou_thresholds is not None:
            # Parameters are already provided; no need to resolve input
            return None
        # --- for SDK call ---

        inputs = types.Object()

        # whole dataset or current view button
        sample_selection = ["entire dataset", "current view"]

        sample_selection_radio_group = types.RadioGroup()
        for selection in sample_selection:
            sample_selection_radio_group.add_choice(selection, label=selection)

        inputs.enum(
            "sample_selection",
            sample_selection_radio_group.values(),
            label="Select samples",
            description="Choose which samples to calculate the IAA for:",
            types=types.RadioView(),
            default="entire dataset"
        )

        dataset = ctx.dataset  # Access the dataset from context

        # Handle sample selection choice
        if ctx.params.get("sample_selection") == "current view":
            dataset = dataset.view()

        # Check available annotation types (bbox, polygon, mask)
        available_types = ctx.params.get("available_types", [])
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

        # Add execution mode (if applicable to your use case)
        _execution_mode(ctx, inputs)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        # Access the dataset
        dataset = ctx.dataset

        # Handle sample selection choice
        if ctx.params.get("sample_selection") == "current view":
            dataset = dataset.view()

        ann_type = ctx.params.get("annotation_type")

        iou_thresholds = ctx.params.get("iou_thresholds")

        if ann_type == "bounding box":
            field_name = "bbox"
        elif ann_type == "mask":
            field_name = "segm"
        elif ann_type == "polygon":
            field_name = "segm"
        else:
            raise Exception(f"Invalid annotation type: {ann_type}")

        # Add the field to the dataset if it does not already exist
        field_path = f"{field_name}_iaa"
        if not dataset.has_sample_field(field_path):
            dataset.add_sample_field(
                field_path,
                fo.DictField,
                subfield=fo.FloatField
            )

        for iou in iou_thresholds:
            iou_str = str(iou).replace(".", ",")
            if not dataset.has_sample_field(f"{field_path}_iou_{iou_str}"):
                dataset.add_sample_field(f"{field_path}_iou_{iou_str}", fo.FloatField)

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

                iaa_dict = sample[field_path]
                if iaa_dict is None:
                    iaa_dict = {}
                iaa_dict[str(iou_threshold)] = alpha
                iou_str = str(iou_threshold).replace(".", ",")
                sample[f"{field_path}_iou_{iou_str}"] = alpha
                sample[field_path] = iaa_dict
                sample.save()
                alphas[str(iou_threshold)].append(alpha)

        message = "Mean K-Alpha for:"
        for iou_threshold in iou_thresholds:
            u_k_alpha = sum(alphas[str(iou_threshold)]) / len(alphas[str(iou_threshold)])
            message += f"\n\tIoU {iou_threshold}: {u_k_alpha}"

        print(message)

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