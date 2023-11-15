import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import urllib.request
import json
import matplotlib.pyplot as plt
import torchvision
from matplotlib import patches
import torchvision.transforms as T
from collections import Counter

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

    # Create output path which is stored in directory 'filtered'
    output_json_path = os.path.join(output_dir, filename)

    # print count values of categories
    category_ids = [annotation["category_id"] for annotation in coco_data["annotations"]]
    print("Count label categories:", Counter(category_ids))


    # Write the new JSON data to a new file
    with open(output_json_path, 'w') as output_file:
        json.dump(coco_data, output_file, indent=2)

    return output_json_path


def download_helper_functions():
    """ Download the helper functions from references files from the PyTorch repository.
    """

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


class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        is_crowd = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]["category_id"])
            is_crowd.append(coco_annotation[i]["iscrowd"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(is_crowd, dtype=torch.int64)

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Annotation is in dictionary format
        my_annotation = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_object_detection(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def plot_img_bbox_target(img, target, filepath):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box, label in zip(target['boxes'], target['labels']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
        # Draw the bounding box on top of the image
        a.add_patch(rect)

        # Display label near the bounding box
        label_str = f"{id2label[label.item()]}"
        a.text(x, y, label_str, fontsize=8, color='r', verticalalignment='top')

    plt.show()

    # save image
    fig.savefig(filepath)


def plot_img_bbox_pred(img, target, filepath, iou_thresh=0.5):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)

    for box, label, score in zip(target['boxes'], target['labels'], target['scores']):
        if score >= iou_thresh:
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')
            # Draw the bounding box on top of the image
            a.add_patch(rect)

            # Display label and score near the bounding box
            label_str = f"{id2label[label.item()]}: {score:.3f}"
            a.text(x, y, label_str, fontsize=8, color='r', verticalalignment='top')

    plt.show()

    # save image
    fig.savefig(filepath)


# torch to PIL
def torch_to_pil(img):
    return T.ToPILImage()(img).convert("RGB")
