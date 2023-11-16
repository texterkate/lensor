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
    download_helper_functions,
    filter_images_with_annotations,
    plot_inference_results,
    save_model,
    log_metrics,
id2label
)

if __name__ == '__main__':
    print("Torch version:", torch.__version__)
    writer = SummaryWriter()
    TRAIN = False

    # download_helper_functions()

    # create train, validation and test datasets
    train_dataset = LensorDataset(root=config.train_img_dir,
                                  annotation=filter_images_with_annotations(config.train_coco),
                                  transforms=get_transform())
    val_dataset = LensorDataset(root=config.val_img_dir, annotation=filter_images_with_annotations(config.val_coco),
                                transforms=get_transform())
    test_dataset = LensorDataset(root=config.test_img_dir, annotation=filter_images_with_annotations(config.test_coco),
                                 transforms=get_transform())

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
        print(f"Start training for {config.num_epochs} epochs...")
        for epoch in range(config.num_epochs):
            # train for one epoch, printing every 10 iterations
            metric_logger = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
            # save model
            save_model(model, epoch)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            coco_evaluator = evaluate(model, val_dataloader, device=device)
            # add metrics to tensorboard
            log_metrics(writer, metric_logger, epoch, coco_evaluator)

    # load best model model
    model = get_model_object_detection(config.num_classes)
    if device == torch.device("cpu"):
        model.load_state_dict(torch.load("best_model/model_epoch_8.pth", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load("best_model/model_epoch_8.pth"))

    # move model to the right device
    model.to(device)

    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in test_dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

            for image, output, target in zip(images, outputs, targets):
                result = {"image": image, "img_id": target['image_id'], "label": target["labels"].numpy(),
                          "prediction": output["labels"].numpy(), "boxes_prediction": output["boxes"].numpy(),
                          "boxes_target": target['boxes'], "scores": output["scores"].numpy()}
                results.append(result)

        for result in results:
            image = result["image"]
            img_id = result["img_id"]
            predictions = [f"{id2label[prediction]}: {score:.3f}" for prediction, score in zip(result["prediction"], result["scores"]) if score >= 0.5]
            targets = [id2label[label] for label in result["label"]]
            boxes_target = result["boxes_target"]
            boxes_prediction = np.array([box for box, score in zip(result["boxes_prediction"], result['scores']) if score >= 0.5])
            writer.add_image_with_boxes(f'Inference/Example_{img_id}_target', image, box_tensor=boxes_target, labels=targets)
            writer.add_image_with_boxes(f'Inference/Example_{img_id}_prediction', image, box_tensor=boxes_prediction, labels=predictions)

    # # evaluate on the test dataset
    # test_evaluator = evaluate(model, test_dataloader, device=device)
    #
    # # add metrics to tensorboard
    # log_metrics(writer, None, 0, test_evaluator)
    #
    # # inference
    # plot_inference_results(model, device, test_dataset, sample_size=1)

    writer.close()
