# Lensor: Damage Detection Challenge


## Prerequisites
Make sure to install all required packages by running the following command in the terminal:

```bash
pip install -r requirements.txt
```

## How to run?
To run the program, run the following command in the terminal. If you want to train the model, add the `--train` flag to the command. 

```bash
python main.py --train
```

Otherwise, the program will load the pre-trained model from the _best_model_ directory and run the inference on the test set.

```bash
python main.py
```

## Model architecture
I prioritize a structured pipeline and systematic logging over extensive model experimentation. 
For object detection, I have opted for a bounding box-based approach, leveraging the Faster R-CNN model with a ResNet-50-FPN backbone, a popular object detection algorithm introduced in the paper: [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

For specific model configurations, such as the number of epochs, batch size, and learning rate, please refer to the _config.py_ file. These settings are customizable for experimenting with different model setups, but I considered such exploration beyond the scope of this assignment.

## Model performance
The model performance can be monitored using Tensorboard. To run Tensorboard, run the following command in the terminal:

```bash
tensorboard --logdir=runs
```

## Results
In the Tensorboard dashboard, you can find the following folders:
- **train_losses**: contains the training loss for each epoch
- **validation_scores**: contains the validation scores for each epoch
- **validation_inference**: contains the inference on sample images from validation set for each epoch
- **test_scores**: contains the test scores
- **test_inference**: contains the inference on sample images from test set



