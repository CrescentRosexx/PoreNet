import os
import argparse
import random
import torch
import wandb
from torch import nn
import torch.optim as optim
import numpy as np
from torch.utils import data
from pathlib import Path
from model import PoreNet
from dataset import FoamAlDataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from predict import evaluate_model
from joblib import load

random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)
torch.cuda.manual_seed(2024)
torch.cuda.manual_seed_all(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Porous Prediction')
parser.add_argument('--lr', type=float, default=0.0031, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--use_att', type=bool, default=True, help='use attention mechanism')
parser.add_argument('--attention_dim', type=int, default=256, help='attention dimension')
parser.add_argument('--label_norm', type=bool, default=True, help='label normalization')
parser.add_argument('--lambda_monotonic', type=float, default=0.1, help='monotonic loss weight')
parser.add_argument('--lambda_local', type=float, default=0.5, help='local loss weight')
parser.add_argument('--drop', type=float, default=0.2, help='drop rate')
parser.add_argument('--jitter', type=float, default=0, help='jitter rate')

parser.add_argument('--result_path', type=str, default='predict_results', help='result path')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--ckpt', type=str, default='ckpt', help='path to checkpoint')
parser.add_argument('--dataset', type=str, default='dataset', help="dataset path")
opt = parser.parse_args()


def create_dataloader(dataset, batch_size, shuffle):
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle)


if __name__ == '__main__':

    wandb.init(project="PoreNet", name='best_run', config={
        "learning_rate": opt.lr,
        "weight_decay": opt.weight_decay,
        "dropout": opt.dropout,
        "lambda_monotonic": opt.lambda_monotonic,
        "lambda_local": opt.lambda_local,
        "drop": opt.drop,
        "jitter": opt.jitter,
    })

    train_dataset = FoamAlDataset(root=Path(opt.dataset), label_norm=opt.label_norm, part='train',
                                  augmentation=True, drop=opt.drop, jitter=opt.jitter)
    test_dataset = FoamAlDataset(root=Path(opt.dataset), label_norm=opt.label_norm, part='test')

    train_loader = create_dataloader(train_dataset, opt.batch_size, True)
    test_loader = create_dataloader(test_dataset, 64, False)

    model = PoreNet(feature_channels=train_dataset.f_channel,
                       num_classes=train_dataset.label_channel,
                       global_channels=train_dataset.c_channel,
                       dropout=opt.dropout, attention_dim=opt.attention_dim)
    # model = PointNetPlusPlus(trans=True, dropout=opt.dropout)
    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))

    print("CUDA:" + str(torch.version.cuda))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=30, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    criterion_local = nn.L1Loss()

    best_pred = -999999999
    for epoch in tqdm(range(opt.nepoch)):
        training_loss = 0.0
        model.train()
        for fine, coarse, targets in train_loader:  # (bs, N, 10)
            fine, coarse, targets = (fine.to(device=device),
                                     coarse.to(device=device),
                                     targets.to(device=device))

            optimizer.zero_grad()
            outputs = model(coarse, fine)

            base_loss = criterion(outputs, targets)
            local_loss = criterion_local(outputs, targets) 
            monotonic_loss = torch.mean(torch.relu(outputs[:, :-1] - outputs[:, 1:]))

            total_loss = base_loss + opt.lambda_monotonic * monotonic_loss + opt.lambda_local * local_loss
            total_loss.backward()
            optimizer.step() 

            training_loss += total_loss.item()

        scheduler.step(training_loss)

        training_loss /= len(train_loader)
        wandb.log({"train_loss": training_loss}, step=epoch)

        # Evaluate on test set
        model.eval() 
        test_outputs = []
        test_targets = []
        test_loss = 0.0
        with torch.no_grad():
            for fine, coarse, targets in test_loader:
                fine, coarse, targets = fine.to(device), coarse.to(device), targets.to(device)

                outputs = model(coarse, fine)

                base_loss = criterion(outputs, targets)
                local_loss = criterion_local(outputs, targets)  
                monotonic_loss = torch.mean(torch.relu(outputs[:, :-1] - outputs[:, 1:])) 

                total_loss = base_loss + opt.lambda_monotonic * monotonic_loss + opt.lambda_local * local_loss
                test_loss += total_loss.item()
                test_outputs.extend(outputs.detach().cpu().numpy())
                test_targets.extend(targets.detach().cpu().numpy())

        test_loss /= len(test_loader) 
        all_outputs = np.array(test_outputs)
        all_targets = np.array(test_targets)
        if opt.label_norm:
            # log normalization
            min_max = load(Path(opt.dataset) / 'min_max_vals.bin')
            min_vals = min_max['min_vals']
            max_vals = min_max['max_vals']
            all_outputs = np.exp(all_outputs * (np.log(max_vals) - np.log(min_vals)) + np.log(min_vals))
            all_targets = np.exp(all_targets * (np.log(max_vals) - np.log(min_vals)) + np.log(min_vals))

        abs_diff = np.abs(all_targets - all_outputs)
        diff_percentage = abs_diff / all_targets
        pred = 1 - diff_percentage
        pre_result = np.mean(pred)

        wandb.log({"test_loss": test_loss, "pred_acc": pre_result}, step=epoch)

        tqdm.write(
            f'Epoch {epoch + 1}/{opt.nepoch} - Train_loss: {training_loss:.4f} - test_loss: {test_loss:.4f} - pred_acc: {pre_result:.4f}')

        ckpt_dir = Path(opt.ckpt)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # save checkpoint
        # torch.save(model.state_dict(), f'{opt.ckpt}/predict_model_{epoch}.pth')

        if pre_result > best_pred:
            best_pred = pre_result
            torch.save(model.state_dict(), f'{opt.ckpt}/best_model.pth')

    model.load_state_dict(torch.load(f'{opt.ckpt}/best_model.pth'))
    evaluate_model(test_loader, model, criterion, device, len(test_dataset), opt.result_path, opt.label_norm)
