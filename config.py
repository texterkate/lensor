DEBUG = True

# directory to images
train_img_dir = "data/images/train"
val_img_dir = "data/images/val"
test_img_dir = "data/images/test"

if DEBUG:
    train_coco = "data/annotations/instances_train_small.json"
    val_coco = "data/annotations/instances_val_small.json"
    test_coco = "data/annotations/instances_test_small.json"
else:
    train_coco = "data/annotations/instances_train.json"
    val_coco = "data/annotations/instances_val.json"
    test_coco = "data/annotations/instances_test.json"

# Batch size
train_batch_size = 2

# Params for dataloader
train_shuffle_dl = False
num_workers_dl = 4

# Params for training
num_classes = 8
num_epochs = 2

lr = 0.005
momentum = 0.9
weight_decay = 0.005