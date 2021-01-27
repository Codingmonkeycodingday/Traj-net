import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from trajnet import PCN
from utils import generate_dataset, load_data, get_normalized_adj, masked_mae_loss
import utils

parser = argparse.ArgumentParser(description='Traj-net')
parser.add_argument('--enable-cuda', action='store_true', default=True,
                    help='Enable CUDA')
parser.add_argument('--model', type=str, default='Traj-net',
                    help='select model')
parser.add_argument('--num_timesteps_input', type=int, default=6,
                    help='input slices')
parser.add_argument('--num_timesteps_output', type=int, default=6,
                    help='output slides')
parser.add_argument('--epochs', type=int, default=0,
                    help='configure epochs')
parser.add_argument('--batch_size', type=int, default=1,
                    help='select model')
parser.add_argument('--pathNum', type=int, default=5,
                    help='select model')
parser.add_argument('--pathLen', type=int, default=7,
                    help='select model')
parser.add_argument('--lr', type=int, default=1e-3,
                    help='select model')

args = parser.parse_args()
args.device = None
num_timesteps_input = args.num_timesteps_input
num_timesteps_output = args.num_timesteps_output
epochs = args.epochs
batch_size = args.batch_size
pathNum = args.pathNum
pathLen = args.pathLen

if args.enable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:0')
else:
    args.device = torch.device('cpu')


def train_epoch(batch_size, P, roots, randomtrajs, mask):
    permutation = torch.randperm(training_input.shape[0])
    curtime = os.times().elapsed
    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        X_batch, X_batch_daily, X_batch_weekly, X_batch_coarse, y_batch = training_input[indices], training_daily_input[
            indices], training_weekly_input[indices], training_coarse_input[indices], training_target[indices]
        mask1 = torch.from_numpy(mask).to(device=args.device)
        X_batch = X_batch.to(device=args.device)
        X_batch_daily = X_batch_daily.to(device=args.device)
        X_batch_weekly = X_batch_weekly.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(P, roots, X_batch, X_batch_daily, X_batch_weekly, r2sDic, s2rDic, randomtrajs, mask1)
        loss = masked_mae_loss(out, y_batch)
        loss.backward()
        optimizer.step()

        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses) / len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(1)

    X, r2sDic, s2rDic, trajDic, keys = load_data(pathNum, pathLen)
    split_line1 = int(X.shape[2] * 0.7)
    split_line2 = int(X.shape[2] * 0.8)
    split_line3 = int(X.shape[2])

    np.save("train_cd.npy", X[:, :, :split_line1])
    np.save("val_cd.npy", X[:, :, split_line1:split_line2])
    np.save("test_cd.npy", X[:, :, split_line2:])
    means = np.mean(X[:, :, :split_line1], axis=(0, 2))
    stds = np.std(X[:, :, :split_line1], axis=(0, 2))
    X = X - means[0]
    X = X / stds[0]
    print(means)
    print(stds)
    print(X.shape)

    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1 - 1008:split_line2]
    test_original_data = X[:, :, split_line2 - 1008:]

    training_input, training_daily_input, training_weekly_input, training_coarse_input, training_target = generate_dataset(
        train_original_data,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output)
    val_input, val_daily_input, val_weekly_input, val_coarse_input, val_target = generate_dataset(val_original_data,
                                                                                                  num_timesteps_input=num_timesteps_input,
                                                                                                  num_timesteps_output=num_timesteps_output)
    test_input, test_daily_input, test_weekly_input, test_coarse_input, test_target = generate_dataset(
        test_original_data,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output)

    print(training_input.shape)
    print(training_target.shape)
    if args.model == 'Traj-net':
        net = PCN(X.shape[0],
                  training_input.shape[3],
                  num_timesteps_input,
                  num_timesteps_output).to(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    for epoch in range(epochs):
        P, roots, randomtrajs, mask = utils.random_trajs(trajDic, s2rDic, keys, pathNum)
        if args.model == 'Traj-net':
            loss = train_epoch(batch_size, P, roots, randomtrajs, mask)
        training_losses.append(loss)
        print("epoch:" + str(epoch))
        print("Training loss: {}".format(training_losses[-1]))
        # Run validation
        test_losses = []
        test_maes = []
        test_maes = []
        test_maes = []
        with torch.no_grad():
            batch_loss = np.zeros(((split_line2 - split_line1) // batch_size + 1, 6))
            batch_maes = np.zeros(((split_line2 - split_line1) // batch_size + 1, 6))
            batch_mape = np.zeros(((split_line2 - split_line1) // batch_size + 1, 6))
            batch_rmse = np.zeros(((split_line2 - split_line1) // batch_size + 1, 6))
            for i in range((split_line2 - split_line1) // batch_size + 1):
                net.eval()
                mini_val_input = val_input[i * batch_size:min((i + 1) * batch_size, split_line2)].to(device=args.device)
                mini_val_input_daily = val_daily_input[i * batch_size:min((i + 1) * batch_size, split_line2)].to(
                    device=args.device)
                mini_val_input_weekly = val_weekly_input[i * batch_size:min((i + 1) * batch_size, split_line2)].to(
                    device=args.device)
                mini_val_target = val_target[i * batch_size:min((i + 1) * batch_size, split_line2)].to(
                    device=args.device)
                if args.model == 'Traj-net':
                    mask1 = torch.from_numpy(mask).to(device=args.device)
                    out = net(P, roots, mini_val_input, mini_val_input_daily, mini_val_input_weekly, r2sDic, s2rDic,
                              randomtrajs, mask1)
                for j in range(out.shape[2]):
                    test_loss = loss_criterion(out[:, :, j], mini_test_target[:, :, j]).to(device="cpu")
                    batch_loss[i, j] = np.asscalar(test_loss.detach().numpy())
                    out_unnormalized = out[:, :, j].detach().cpu().numpy() * stds[0] + means[0]
                    target_unnormalized = mini_test_target[:, :, j].detach().cpu().numpy() * stds[0] + means[0]
                    mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))

                    batch_rmse[i, j] = utils.masked_rmse_np(out_unnormalized, target_unnormalized)
                    batch_mape[i, j] = utils.masked_mape_np(target_unnormalized, out_unnormalized)
                    batch_maes[i, j] = mae
            print("test loss: {}".format(batch_loss.mean(axis=0)))
            print("test MAE: {}".format(batch_maes.mean(axis=0)))
            print("test MAPE: {}".format(batch_mape.mean(axis=0)))
            print("test RMSE: {}".format(batch_rmse.mean(axis=0)))

    # Run test
    test_losses = []
    test_maes = []
    with torch.no_grad():
        P, roots, randomtrajs, mask = utils.random_trajs(trajDic, s2rDic, keys, pathNum)
        batch_loss = np.zeros(((split_line3 - split_line2) // batch_size + 1, 6))
        batch_maes = np.zeros(((split_line3 - split_line2) // batch_size + 1, 6))
        for i in range((split_line3 - split_line2) // batch_size + 1):
            net.eval()
            mini_test_input = test_input[i * batch_size:min((i + 1) * batch_size, split_line3)].to(device=args.device)
            mini_test_input_daily = test_daily_input[i * batch_size:min((i + 1) * batch_size, split_line3)].to(
                device=args.device)
            mini_test_input_weekly = test_weekly_input[i * batch_size:min((i + 1) * batch_size, split_line3)].to(
                device=args.device)
            mini_test_target = test_target[i * batch_size:min((i + 1) * batch_size, split_line3)].to(device=args.device)
            if args.model == 'Traj-net':
                mask1 = torch.from_numpy(mask).to(device=args.device)
                out = net(P, roots, mini_test_input, mini_test_input_daily, mini_test_input_weekly, r2sDic, s2rDic,
                          randomtrajs, mask1)
            for j in range(out.shape[2]):
                test_loss = loss_criterion(out[:, :, j], mini_test_target[:, :, j]).to(device="cpu")
                batch_loss[i, j] = np.asscalar(test_loss.detach().numpy())
                out_unnormalized = out[:, :, j].detach().cpu().numpy() * stds[0] + means[0]
                target_unnormalized = mini_test_target[:, :, j].detach().cpu().numpy() * stds[0] + means[0]
                mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                batch_maes[i, j] = mae
        print("Validation loss: {}".format(batch_loss.mean(axis=0)))
        print("Validation MAE: {}".format(batch_maes.mean(axis=0)))

    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    with open("checkpoints/losses.pk", "wb") as fd:
        pk.dump((training_losses, validation_losses, validation_maes), fd)

