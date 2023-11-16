import argparse
import numpy as np
import torch
from engine import train_one_epoch, evaluate
import config
from tensorboardX import SummaryWriter
from model_utils import (
    get_model_object_detection,
    collate_fn,
    get_transform,
    LensorDataset,
    filter_images_with_annotations,
    save_model,
    log_metrics,
    log_inference_results,
)


def main(train):

    print("Start execution of main.py")

    # create tensorboard writer
    writer = SummaryWriter()

    # create train, validation and test datasets
    train_dataset = LensorDataset(
        root=config.train_img_dir,
        annotation=filter_images_with_annotations(config.train_coco),
        transforms=get_transform(train=True)
    )
    val_dataset = LensorDataset(
        root=config.val_img_dir,
        annotation=filter_images_with_annotations(config.val_coco),
        transforms=get_transform(train=False)
    )
    test_dataset = LensorDataset(
        root=config.test_img_dir,
        annotation=filter_images_with_annotations(config.test_coco),
        transforms=get_transform(train=False)
    )

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

    if train:

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

        # keep track of best mAP
        max_map = -np.Inf

        # train the model for nr of epochs specified in config
        print(f"Start training for {config.num_epochs} epochs...")
        for epoch in range(config.num_epochs):

            # train for one epoch, printing every 10 iterations
            metric_logger = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)

            # update the learning rate
            lr_scheduler.step()

            # evaluate on the validation dataset
            coco_evaluator = evaluate(model, val_dataloader, device=device)

            # current mAP (iou = 0.5)
            map = coco_evaluator.coco_eval['bbox'].stats[1]

            # save model
            save_model(model, epoch, is_best=map>max_map)
            max_map = max(map, max_map)

            # add metrics to tensorboard
            log_metrics(writer, metric_logger, epoch, coco_evaluator)

            # log inference results
            log_inference_results(writer, model, device, val_dataset, sample_size=config.nr_samples_val, stage="validation", epoch=epoch)

    # load best model
    model = get_model_object_detection(config.num_classes)
    if device == torch.device("cpu"):
        model.load_state_dict(torch.load("best_model/best_model.pth", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("best_model/best_model.pth"))

    # move model to the right device
    model.to(device)

    # set model to evaluation mode
    model.eval()

    # evaluate on the test dataset
    test_evaluator = evaluate(model, test_dataloader, device=device)

    # add test metrics to tensorboard
    log_metrics(writer, None, 0, test_evaluator)

    # log inference results on test set
    log_inference_results(writer, model, device, test_dataset, sample_size=config.nr_samples_test, stage="test")

    # close tensorboard writer
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection Training, Validation and Testing Script')
    parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train the model')
    args = parser.parse_args()
    main(train=args.train)
