
# AutoRec Implementation

This repository contains a PyTorch implementation of the paper **"AutoRec: Autoencoders Meet Collaborative Filtering"**. It is a model designed for collaborative filtering tasks in recommendation systems.

## Features

- **User-based and Item-based AutoRec Training**: Select between user-based or item-based approaches.
- **Customizable Parameters**: Easily configure hidden space dimensions, dropout rates, and other training parameters.
- **Simple Execution**: Just one command to run the model.

## Requirements

Before running the code, ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Running the Code

To train the AutoRec model, simply run the following command:

```bash
python main.py
```

### Arguments
The `main.py` script accepts the following arguments:

| Argument          | Description                                         | Default Value      |
|-------------------|-----------------------------------------------------|--------------------|
| `-u`              | User-based training (default is item-based)         | `False`            |
| `--data_folder`   | Path to the dataset folder                          | `"ml-1m/"`         |
| `--train_ratio`   | Ratio of the dataset to use for training            | `0.9`              |
| `--early_stopping`| Whether to use early stopping                       | `True`             |
| `--epochs`        | Number of training epochs                           | `250`              |
| `--batch_size`    | Batch size for training                             | `512`              |
| `--hsd`           | Hidden space dimension                              | `500`              |
| `--dropout_rate`  | Dropout rate for the AutoRec model                  | `0.4`              |
| `--lr`            | Learning rate for the optimizer                     | `0.007`            |
| `--weight_decay`  | Weight decay for the optimizer                      | `8e-4`             |


## Dataset

The default dataset used is **MovieLens 1M**. Make sure the dataset is located in the folder specified by `--data_folder`. By default, this is `ml-1m/`.

If you want to use another dataset, place it in the desired folder and update the `--data_folder` argument.

## Output

The model's performance is logged in real-time during training, showing metrics like RMSE on the training and validation datasets. The best model weights are automatically saved in a file named:

```plaintext
best_model_hsd<hidden_space_dimension>.pth
```

For example: `best_model_hsd500.pth`

## Reference

If you use this implementation in your research, please cite the original paper:

> Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. **"AutoRec: Autoencoders Meet Collaborative Filtering."** In Proceedings of the 24th International Conference on World Wide Web (WWW), 2015.