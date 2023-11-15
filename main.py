import time
import os
import torch
import matplotlib.pyplot as plt
import torchvision
from matplotlib import patches
import torchvision.transforms as T
from torchmetrics.detection import MeanAveragePrecision
from torchvision.utils import draw_bounding_boxes

import transforms
from engine import train_one_epoch, evaluate
import config
from model_utils import (
    get_model_object_detection,
    collate_fn,
    get_transform,
    myOwnDataset,
    download_helper_functions,
    filter_images_with_annotations,
    plot_img_bbox_target,
    plot_img_bbox_pred,
    torch_to_pil,
)

if __name__ == '__main__':
    print("Torch version:", torch.__version__)

    TRAIN = False

    # download_helper_functions()


    # create train, validation and test datasets
    train_dataset = myOwnDataset(root=config.train_img_dir, annotation=filter_images_with_annotations(config.train_coco), transforms=get_transform())
    val_dataset = myOwnDataset(root=config.val_img_dir, annotation=filter_images_with_annotations(config.val_coco), transforms=get_transform())
    test_dataset = myOwnDataset(root=config.test_img_dir, annotation=filter_images_with_annotations(config.test_coco), transforms=get_transform())


    # define training, validation and test data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=config.train_shuffle_dl,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers_dl,
        collate_fn=collate_fn,
    )

    # select device (whether GPU or CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    if TRAIN:

        # get the model
        model = get_model_object_detection(config.num_classes)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        # train the model for nr of epochs specified in config
        for epoch in range(config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
            # save model
            os.makedirs("saved_models", exist_ok=True)
            torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch}.pth")
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, val_dataloader, device=device)

    else:
        # load model
        model = get_model_object_detection(config.num_classes)
        if device == torch.device("cpu"):
            model.load_state_dict(torch.load("best_model/model_epoch_2.pth", map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load("best_model/model_epoch_2.pth"))
        model.to(device)
        # evaluate(model, test_dataloader, device=device)

        # inference

        preds = []
        targets = []

        for i, item in enumerate(test_dataset):
            if i == 25:
                break
            print(f"Image {i+1} of {len(test_dataset)}")
            img, target = item
            model.eval()
            with torch.no_grad():
                pred = model([img.to(device)])[0]
                preds.append(pred)
                targets.append(target)


        # plot_img_bbox_target(torch_to_pil(img), target)
        # plot_img_bbox_pred(torch_to_pil(img), pred, iou_thresh=0.5)

        # calculate eval scores per class
        metric = MeanAveragePrecision()
        metric.update([preds], [targets])
        print(metric.compute())

        metric_per_class = MeanAveragePrecision(class_metrics=True)
        metric_per_class.update([preds], [targets])
        print(metric_per_class.compute())

        # pred_labels = [f"{label}: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
        # pred_boxes = pred["boxes"].long()
        # img_uint8 = (img * 255).to(torch.uint8)
        # output_image = draw_bounding_boxes(img_uint8, pred_boxes, pred_labels, colors="red")
        # plt.imshow(output_image)


        # plot predictions





    #########################################

    # #
    # n_batches = len(train_dataloader)
    # metric = MeanAveragePrecision()
    # train_losses = []
    # val_scores = []
    #
    # # Training
    # for epoch in range(config.num_epochs):
    #     print(f"Starting epoch {epoch + 1} of {config.num_epochs}")
    #     model.train()
    #     i = 0
    #     time_start = time.time()
    #     running_loss = 0.0
    #     for imgs, annotations in train_dataloader:
    #         i += 1
    #         imgs = list(img.to(device) for img in imgs)
    #         annotations = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in annotations]
    #         loss_dict = model(imgs, annotations)
    #         loss = sum(loss for loss in loss_dict.values())
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #         if i % 2 == 0:
    #             print(f"    [Batch {i:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}.")
    #
    #     train_loss = running_loss / n_batches
    #     train_losses.append(train_loss)
    #
    #     # save model in saved_models folder (create folder if it does not exist)
    #     os.makedirs("saved_models", exist_ok=True)
    #     torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch}.pth")
    #
    #
    #     elapsed = time.time() - time_start
    #     prefix = f"[Epoch {epoch + 1:2d} / {config.num_epochs:2d}]"
    #     print(f"{prefix} Train loss: {train_loss:7.3f} [{elapsed:.0f} secs]", end=' | ')
    #
    #
    #     model.eval()
    #     preds = []
    #     targets = []
    #     cpu_device = torch.device("cpu")
    #     with torch.no_grad():
    #         for imgs, annotations in val_dataloader:
    #             imgs = list(img.to(device) for img in imgs)
    #             annotations = [{k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in annotations]
    #             targets.extend(annotations)
    #             outputs = model(imgs)
    #             outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    #             preds.extend(outputs)
    #
    #         metric.update(preds, targets)
    #         map = metric.compute()
    #         val_scores.append(map['map'])
    #         print(f"Val mAP: {map['map']}")