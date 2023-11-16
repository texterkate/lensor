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



