import time
import os
import torch
from torchmetrics.detection import MeanAveragePrecision
import config
from model_utils import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    myOwnDataset,
    download_helper_functions,
    filter_images_with_annotations
)

if __name__ == '__main__':
    print("Torch version:", torch.__version__)

    # download_helper_functions()
    # from engine import train_one_epoch, evaluate

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

    # get the model
    model = get_model_instance_segmentation(config.num_classes)

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
    #
    # for epoch in range(config.num_epochs):
    #     # train for one epoch, printing every 10 iterations
    #     train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=1)
    #     # update the learning rate
    #     lr_scheduler.step()
    #     # evaluate on the test dataset
    #     evaluate(model, val_dataloader, device=device)
    #
    # print("That's it!")
    #
    n_batches = len(train_dataloader)
    metric = MeanAveragePrecision()
    train_losses = []
    val_scores = []

    # Training
    for epoch in range(config.num_epochs):
        print(f"Starting epoch {epoch + 1} of {config.num_epochs}")
        model.train()
        i = 0
        time_start = time.time()
        running_loss = 0.0
        for imgs, annotations in train_dataloader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in annotations]
            loss_dict = model(imgs, annotations)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2 == 0:
                print(f"    [Batch {i:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}.")

        train_loss = running_loss / n_batches
        train_losses.append(train_loss)

        # save model in saved_models folder (create folder if it does not exist)
        os.makedirs("saved_models", exist_ok=True)
        torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch}.pth")


        elapsed = time.time() - time_start
        prefix = f"[Epoch {epoch + 1:2d} / {config.num_epochs:2d}]"
        print(f"{prefix} Train loss: {train_loss:7.3f} [{elapsed:.0f} secs]", end=' | ')


        model.eval()
        preds = []
        targets = []
        cpu_device = torch.device("cpu")
        with torch.no_grad():
            for imgs, annotations in val_dataloader:
                imgs = list(img.to(device) for img in imgs)
                annotations = [{k: v.to(cpu_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in annotations]
                targets.extend(annotations)
                outputs = model(imgs)
                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                preds.extend(outputs)

            metric.update(preds, targets)
            map = metric.compute()
            val_scores.append(map['map'])
            print(f"Val mAP: {map['map']}")