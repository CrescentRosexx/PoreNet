import os
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
from joblib import load

random.seed(2024)
torch.manual_seed(2024)
np.random.seed(2024)
torch.cuda.manual_seed(2024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_dataloader(dataset, batch_size, shuffle):
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle)


def train():
    wandb.init()
    config = wandb.config

    random.seed(2024)
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.cuda.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset = FoamAlDataset(root=Path(config.dataset), label_norm=config.label_norm, part='train',
                                  augmentation=True, drop=config.drop, jitter=config.jitter, outlier_ratio=config.outlier_ratio)
    test_dataset = FoamAlDataset(root=Path(config.dataset), label_norm=config.label_norm, part='test')

    train_loader = create_dataloader(train_dataset, config.batch_size, True)
    test_loader = create_dataloader(test_dataset, 64, False)

    model = PoreNet(feature_channels=train_dataset.f_channel,
                       num_classes=train_dataset.label_channel,
                       global_channels=train_dataset.c_channel,
                       dropout=config.dropout,
                       attention_dim=256)
    if config.model != '':
        model.load_state_dict(torch.load(config.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=30, verbose=True)
    criterion = nn.MSELoss()
    criterion_local = nn.L1Loss()

    best_pred = -999999999
    for epoch in tqdm(range(config.nepoch)):
        training_loss = 0.0
        model.train()
        for fine, coarse, targets in train_loader:
            fine, coarse, targets = (fine.to(device=device),
                                     coarse.to(device=device),
                                     targets.to(device=device))

            optimizer.zero_grad()
            outputs = model(coarse, fine)

            base_loss = criterion(outputs, targets)
            local_loss = criterion_local(outputs, targets)
            monotonic_loss = torch.mean(torch.relu(outputs[:, :-1] - outputs[:, 1:]))

            total_loss = base_loss + config.lambda_monotonic * monotonic_loss + config.lambda_local * local_loss
            total_loss.backward()
            optimizer.step()

            training_loss += total_loss.item()

        scheduler.step(training_loss)
        training_loss /= len(train_loader)
        wandb.log({"train_loss": training_loss}, step=epoch)

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

                total_loss = base_loss + config.lambda_monotonic * monotonic_loss + config.lambda_local * local_loss
                test_loss += total_loss.item()
                test_outputs.extend(outputs.detach().cpu().numpy())
                test_targets.extend(targets.detach().cpu().numpy())

        test_loss /= len(test_loader)
        all_outputs = np.array(test_outputs)
        all_targets = np.array(test_targets)
        if config.label_norm:
            min_max = load(Path(config.dataset) / 'min_max_vals.bin')
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
            f'Epoch {epoch + 1}/{config.nepoch} - Train_loss: {training_loss:.4f} - test_loss: {test_loss:.4f} - pred_acc: {pre_result:.4f}')

        if pre_result > best_pred:
            best_pred = pre_result
            wandb.log({"best_pred": best_pred}, step=epoch)
            if best_pred > 0.949:
                torch.save(model.state_dict(), f'{config.ckpt}/best_model_{best_pred:.4f}.pth')

    # model.load_state_dict(torch.load(f'{config.ckpt}/best_model.pth'))
    # evaluate_model(test_loader, model, criterion, device, len(test_dataset), config.result_path, config.label_norm)


def main():
    # sweep_config = {
    #     'method': 'random',
    #     'metric': {
    #         'name': 'best_pred',
    #         'goal': 'maximize'
    #     },
    #     'parameters': {
    #         'lr': {
    #             'distribution': 'q_uniform',
    #             'min': 0.0015,
    #             'max': 0.004,
    #             'q': 0.0001
    #         },
    #         'batch_size': {'value': 32},
    #         'dropout': {'values': [0.2, 0.3, 0.4, 0.5]},
    #         'weight_decay': {'values': [0.0005, 0.001]},
    #         'lambda_monotonic': {'values': [0.1, 0.2, 0.3]},
    #         'lambda_local': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    #         'drop': {'values': [0, 0.1, 0.2, 0.3]},
    #         'jitter': {'value': 0},
    #         'nepoch': {'value': 500},
    #         'label_norm': {'value': True},
    #         'result_path': {'value': 'predict_results'},
    #         'model': {'value': ''},
    #         'ckpt': {'value': 'ckpt'},
    #         'dataset': {'value': 'dataset'}
    #     }
    # }

    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'best_pred',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {'value': 0.0031},
            'batch_size': {'value': 32},
            'dropout': {'value': 0.3},
            'weight_decay': {'value': 0.001},
            'attention_dim': {'value': 256},
            'lambda_monotonic': {'value': 0.2},
            'lambda_local': {'value': 0.5},
            'drop': {'value': 0.2},
            # 'jitter': {'values': [round(x * 0.01, 2) for x in range(21, 41)]},
            'jitter': {'value': 0},
            'outlier_ratio': {'values': [round(x * 0.1, 2) for x in range(1, 9)]},
            'nepoch': {'value': 500},
            'label_norm': {'value': True},
            'result_path': {'value': 'predict_results'},
            'model': {'value': ''},
            'ckpt': {'value': 'ckpt'},
            'dataset': {'value': 'dataset'}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="FoamSandwich-sweep-robust")
    wandb.agent(sweep_id, train)


if __name__ == '__main__':
    main()
