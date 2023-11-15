import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import urllib.request
import json

# # Path to the input JSON file
# input_file_path = "data/annotations/instances_test.json"
#
# # Path to the output JSON file
# output_file_path = "data/annotations/instances_test_small.json"
#
# # Load the JSON file
# with open(input_file_path, 'r') as json_file:
#     data = json.load(json_file)
#
# # Keep only 'info', 'licenses', and 'categories'
# new_data = {
#     'info': data.get('info', {}),
#     'licenses': data.get('licenses', []),
#     'categories': data.get('categories', [])
# }
#
# # Keep only the first 20 items from 'images' and 'annotations'
# new_data['images'] = data.get('images', [])[:5]
# new_data['annotations'] = data.get('annotations', [])[:5]
#
# # Write the modified data to a new JSON file
# with open(output_file_path, 'w') as new_json_file:
#     json.dump(new_data, new_json_file, indent=2)
#
# print(f"Data from '{input_file_path}' has been processed and saved to '{output_file_path}'.")



import json

def filter_images_with_annotations(coco_json_path):

    # Get filename incl extension
    filename = os.path.basename(coco_json_path)

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
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        # img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

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

# def get_model_instance_segmentation(num_classes):
#     # load an instance segmentation model pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
#
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(
#         in_features_mask,
#         hidden_layer,
#         num_classes,
#     )
#
#     return model