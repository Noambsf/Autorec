import torch
import argparse

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from copy import deepcopy
from utils import *

def main() :
    parser = argparse.ArgumentParser(description="AutoRec")

    parser.add_argument("--data_folder", type=str, default="ml-1m/")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--early_stopping", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hsd", type=int, default=500, help="Hidden Space Dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.007)
    parser.add_argument("--weight_decay", type=float, default=8e-4)

    parser.add_argument("-u", type=bool, default=False, help="User-based training (default is Item-based)")


    args = parser.parse_args()
    data_folder = args.data_folder


    num_users = 6040
    num_movies = 3952
    ratings_full, _ = load_ratings(data_folder, num_users, num_movies)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    user_based = False
    # User-based training
    if user_based : 
        input_dim = num_movies
    # Item-based training
    else : 
        input_dim = num_users
        ratings_full = ratings_full.T

    # Get training and testing matrices
    new_train_ratings, new_test_ratings = split_matrix(ratings_full, ratio=args.train_ratio)
    new_train_ratings, new_test_ratings = torch.Tensor(new_train_ratings), torch.Tensor(new_test_ratings)

    new_train_ratings = new_train_ratings.masked_fill(new_train_ratings==0, float('nan'))
    new_test_ratings = new_test_ratings.masked_fill(new_test_ratings==0, float('nan'))

    new_train_ratings, new_train_mask = get_mask(new_train_ratings)
    new_test_ratings, new_test_mask = get_mask(new_test_ratings)


    nb_epochs = args.epochs
    batch_size = args.batch_size
    hidden_space_dimension = args.hsd
    dropout_rate = args.dropout_rate

    best_rmse = 1e5
    best_model = None
    early_stopping = args.early_stopping

    train_dataset = TensorDataset(new_train_ratings, new_train_mask)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    model = AutoRec(input_dim, hidden_space_dimension, dropout_rate=dropout_rate)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                mode='min',
                                factor=0.5, 
                                patience=5)
    model = model.to(device)
    pbar = tqdm(range(nb_epochs))

    # Main training loop
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for batch_ratings, batch_mask in train_loader:
            batch_ratings = batch_ratings.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            output_matrix = model(batch_ratings)
            loss = compute_mse(output_matrix, batch_ratings, batch_mask)
            loss.backward()
            optimizer.step()

            output_matrix = adjust_outliers(output_matrix, 1, 5)
            epoch_loss += compute_rmse(output_matrix, batch_ratings.to(device), batch_mask.to(device)).item()
            
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)

        # Early stopping using validation data
        if early_stopping :
            model.eval()
            
            with torch.no_grad() :
                predictions = model(new_train_ratings.to(device))
                predictions = adjust_outliers(predictions, 1, 5)
                test_loss = compute_rmse(predictions, new_test_ratings.to(device), new_test_mask.to(device))
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_description(
                    f"Hidden_space_dimension: {hidden_space_dimension}. Epoch {epoch + 1}, "
                    f"RMSE on Training data: {epoch_loss:.4f}, "
                    f"Validation RMSE: {test_loss.item():.4f}, "
                    f"Learning Rate: {current_lr}"
                )
                if test_loss.item() < best_rmse :
                    best_rmse = test_loss
                    best_model = deepcopy(model)
                    saved_weights_epoch = epoch + 1

    
    torch.save(best_model.state_dict(), f"best_model_hsd{hidden_space_dimension}.pth")
    print(f"Saved weights at epoch : {saved_weights_epoch} and got for Validaiton/Test rmse : {best_rmse}")

if __name__== "__main__" :
    main()
