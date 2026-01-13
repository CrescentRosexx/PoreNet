import argparse
import random
import torch
import os
from torch import nn
import numpy as np
from torch.utils import data
from pathlib import Path
from dataset import FoamAlDataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from joblib import load
from model import PoreNet

parser = argparse.ArgumentParser(description='PoreNet Prediction')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--label_norm', type=bool, default=True, help='label normalization')
parser.add_argument('--model', type=str, default='ckpt/best_model.pth', help='model path')
parser.add_argument('--dataset', type=str, default='dataset', help="dataset path")
parser.add_argument('--result_path', type=str, default='test_results', help='result path')
parser.add_argument('--lambda_monotonic', type=float, default=0.1, help='monotonic loss weight')
opt = parser.parse_args()


def evaluate_model(test_loader, model, criterion, device, test_size, result_path, label_norm=True):
    if not Path(result_path).exists():
        Path(result_path).mkdir()
    model.eval()
    test_loss = 0.0
    all_outputs = []
    all_targets = []
    print("Evaluating model...")

    with torch.no_grad():
        for fine, coarse, target in test_loader:  # (bs, N, 10)
            fine, coarse, target = (fine.to(device=device),
                                    coarse.to(device=device),
                                    target.to(device=device))

            outputs = model(coarse, fine)
            base_loss = criterion(outputs, target)
            monotonic_loss = torch.mean(torch.relu(outputs[:, :-1] - outputs[:, 1:]))

            total_loss = base_loss + opt.lambda_monotonic * monotonic_loss
            test_loss += total_loss.item()
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)

    x = np.linspace(0, 0.6, 51)

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    if label_norm:
        min_max = load(Path(opt.dataset) / 'min_max_vals.bin')
        min_vals = min_max['min_vals']
        max_vals = min_max['max_vals']
        all_outputs = np.exp(all_outputs * (np.log(max_vals) - np.log(min_vals)) + np.log(min_vals))
        all_targets = np.exp(all_targets * (np.log(max_vals) - np.log(min_vals)) + np.log(min_vals))

    test_mse = mean_squared_error(all_targets, all_outputs) 

    abs_diff = np.abs(all_targets - all_outputs)
    diff_percentage = abs_diff / all_targets
    pred = 1 - diff_percentage
    pre_result = np.mean(pred)
    per_predict = np.mean(pred, axis=1)

    per_mape = np.mean(diff_percentage, axis=1) 
    per_mae = np.mean(abs_diff, axis=1)

    mae = np.mean(per_mae)

    print(f'MAE: {mae:.4f}')

    print(
        f'Test Loss: {avg_test_loss:.4f}, Test MSE: {test_mse:.4f} - Accuracy: {pre_result:.4f}')
    
    # Save detailed results
    import pandas as pd
    files = os.listdir(os.path.join(opt.dataset, 'test/energy'))
    global_pd = pd.read_csv(os.path.join(opt.dataset, 'test/global.csv'))
    new_pd = pd.DataFrame(columns=['file', 'predict', 'length', 'volume fraction'])
    for file, predict in zip(files, per_predict):
        local = np.array(pd.read_csv(os.path.join(opt.dataset, 'test/local', file)).values, dtype=np.float32)
        length = local.shape[0]
        volume_fraction = global_pd[global_pd['name'] == file.split('.')[0]]['volume_fraction'].values[0]
        new_pd.loc[len(new_pd)] = [file, predict, length, volume_fraction]

    new_pd.to_csv(f'{result_path}/predict_results.csv', index=False)

    new_pd['mae'] = per_mae
    new_pd['mape'] = per_mape
    new_pd.to_csv(f'{result_path}/mae_mape.csv', index=False)

    os.makedirs(f"{result_path}/results", exist_ok=True)
    print("Plotting results...")
    for (file, output, target, predict) in tqdm(zip(files, all_outputs, all_targets, per_predict)):
        name = file.split('.')[0]
        output = np.insert(output, 0, [0])
        target = np.insert(target, 0, [0])

        pd.DataFrame({'output': output, 'target': target}).to_csv(f'{result_path}/results/{name}_{predict}.csv', index=False)
    
        plt.figure()
        plt.xlabel('Strain')
        plt.ylabel('Energy')
        plt.title(name + f' {predict:.4f}')
        plt.plot(x, output, label='Predicted')
        plt.plot(x, target, label='Target')
        plt.legend()
        plt.savefig(f'{result_path}/{name}.png')
        plt.close()

if __name__ == '__main__':

    test_dataset = FoamAlDataset(root=Path(opt.dataset), label_norm=opt.label_norm, part='test')

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PoreNet(feature_channels=test_dataset.f_channel,
                       num_classes=test_dataset.label_channel,
                       global_channels=test_dataset.c_channel)
    if opt.model:
        model.load_state_dict(torch.load(opt.model))
    else:
        raise ValueError('Model not found!')
    model.to(device=device)

    criterion = nn.MSELoss()
    result_path = opt.result_path
    evaluate_model(test_loader, model, criterion, device, len(test_dataset), result_path, opt.label_norm)

# MAE: 0.0174
# Test Loss: 0.0001, Test MSE: 0.0008 - Accuracy: 0.9512
