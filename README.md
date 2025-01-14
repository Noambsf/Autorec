
# AutoRec Implementation

This repository contains a PyTorch implementation of the paper **"AutoRec: Autoencoders Meet Collaborative Filtering"**. It is a model designed for collaborative filtering tasks in recommendation systems.


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

## Reference

If you use this implementation in your research, please cite the original paper:

1. Sedhain, S., Menon, A. K., Sanner, S., & Xie, L. (2015). AutoRec: Autoencoders Meet Collaborative Filtering. In *Proceedings of the 24th International Conference on World Wide Web* (pp. 111–112). Association for Computing Machinery. DOI: [10.1145/2740908.2742726](https://doi.org/10.1145/2740908.2742726)