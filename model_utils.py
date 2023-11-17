import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import urllib.request
import json
import torchvision
from collections import Counter
import config
from utils import MetricLogger
from tensorboardX import SummaryWriter
from coco_eval import CocoEvaluator
import torchvision.transforms as T

id2label = {
    0: 'severity-damage',
    1: 'minor-dent',
    2: 'minor-scratch',
    3: 'moderate-broken',
    4: 'moderate-dent',
    5: 'moderate-scratch',
    6: 'severe-broken',
    7: 'severe-dent',
    8: 'severe-scratch'
}


def filter_images_with_annotations(coco_json_path):
    """Filter images in a COCO JSON file to keep only images with at least one annotation."""

    # Get filename incl extension
    filename = os.path.basename(coco_json_path)
    print("Reading file:", filename)

    # Read the COCO JSON file
    with open(coco_json_path, 'r') as coco_file:
        coco_data = json.load(coco_file)

    # Extract image IDs with annotations
    annotated_image_ids = set(annotation['image_id'] for annotation in coco_data['annotations'])

    # Filter images in the COCO data based on annotations
    filtered_images = [img for img in coco_data['images'] if img['id'] in annotated_image_ids]

    # Update the images field in the JSON data
    coco_data['images'] = filtered_images

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(coco_json_path), 'cleaned')
    os.makedirs(output_dir, exist_ok=True)

    # Create output path of new COCO JSON file
    output_json_path = os.path.join(output_dir, filename)

    # print count values of categories (to investigate class imbalance)
    category_ids = [annotation["category_id"] for annotation in coco_data["annotations"]]
    print("Count label categories:", Counter(category_ids))

    # Write the new JSON data to a new file
    with open(output_json_path, 'w') as output_file:
        json.dump(coco_data, output_file, indent=2)

    return output_json_path


def download_helper_functions():
    """ Download the helper functions from references files from the PyTorch repository."""

    # URLs of the files to download
    urls = [
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
        "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py",
    ]

    # Download each file and save it in the root directory
    for url in urls:
        file_name = url.split("/")[-1]  # Extracting the file name from the URL
        file_path = os.path.join(os.getcwd(), file_name)
        urllib.request.urlretrieve(url, file_path)

    print("Files with helper functions downloaded and saved in the root directory.")


class LensorDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]
        # Get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # Target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)
        # Path for input image
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        # Open the input image
        img = Image.open(os.path.join(self.root, path))
        # Bounding boxes for objects (transform to [xmin, ymin, xmax, ymax] format)
        boxes = torch.tensor(
            [[anno["bbox"][0], anno["bbox"][1], anno["bbox"][0] + anno["bbox"][2], anno["bbox"][1] + anno["bbox"][3]]
             for anno in coco_annotation], dtype=torch.float32)
        # Labels (8 classes in total)
        labels = torch.tensor([anno["category_id"] for anno in coco_annotation], dtype=torch.int64)
        # Is crowd
        is_crowd = torch.tensor([anno["iscrowd"] for anno in coco_annotation], dtype=torch.int64)
        # Size of bbox (Rectangular)
        areas = torch.tensor([anno["area"] for anno in coco_annotation], dtype=torch.float32)
        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": is_crowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    """Get a list of transformations for the train/test datasets."""
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_object_detection(num_classes):
    """Get a pre-trained object detection model from torchvision."""

    # load an object detection model (pre-trained)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def save_model(model, epoch, is_best=False):
    """Save model checkpoint to disk."""

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch}.pth")

    if is_best:
        torch.save(model.state_dict(), f"best_model/best_model.pth")


coco_metrics = [
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
]


def log_metrics(writer: SummaryWriter, metric_logger: MetricLogger, epoch: int,
                coco_evaluator: CocoEvaluator) -> None:
    """Log metrics (train loss and validation mAP) to TensorBoard."""

    is_train = metric_logger is not None

    # log train loss
    if is_train:
        for k, v in metric_logger.meters.items():
            writer.add_scalar(f"train_losses/{k}", v.median, epoch)

    # log val/test mAP
    for i, metric in enumerate(coco_metrics):
        metric = (f"validation_scores/{metric}" if is_train else f"test_scores/{metric}")
        writer.add_scalar(metric, coco_evaluator.coco_eval['bbox'].stats[i], epoch)


def log_inference_results(writer: SummaryWriter, model, device, dataset, sample_size, stage="validation",
                          epoch="") -> None:
    """Log inference results (both prediction and target images) to TensorBoard."""

    # Take sample images from dataset
    sampler = data.RandomSampler(dataset, num_samples=sample_size, generator=torch.Generator().manual_seed(42))
    dataloader = data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=False,
                                 num_workers=config.num_workers_dl,
                                 collate_fn=collate_fn, sampler=sampler)

    # Log inference results (target and prediction) for each image to TensorBoard
    with torch.no_grad():
        for images, targets in dataloader:

            # Move images and targets to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to("cpu") if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            # Get model predictions
            outputs = model(images)
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            # Log inference results for each image
            for image, output, target in zip(images, outputs, targets):

                # Get image ID, labels, predictions, boxes, and scores
                img_id = target['image_id']
                labels = target["labels"].numpy()
                predictions = output["labels"].numpy()
                boxes_target = target['boxes'].numpy()
                boxes_prediction = output["boxes"].numpy()
                scores = output["scores"].numpy()

                # Filter predictions with scores >= 0.5
                high_score_indices = scores >= 0.5
                predictions = predictions[high_score_indices]
                scores = scores[high_score_indices]

                # Convert label indices to label names
                targets = [id2label[label] for label in labels]
                predictions = [f"{id2label[prediction]}: {score:.3f}" for prediction, score in zip(predictions, scores)]

                # Add images with boxes to TensorBoard
                writer.add_image_with_boxes(f'{stage}_inference/Example_{img_id}_target_{epoch}', image,
                                            box_tensor=boxes_target, labels=targets)
                writer.add_image_with_boxes(f'{stage}_inference/Example_{img_id}_prediction_{epoch}', image,
                                            box_tensor=boxes_prediction[high_score_indices], labels=predictions)
