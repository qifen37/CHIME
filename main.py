import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model.CHIME import CHIME
from utils.data_loader import load_data, get_train_test
from train_test import train_one_epoch, test
from utils.EarlyStopping import EarlyStopping
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def run(args):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    data_path = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name)
    kg_g, smiles = load_data(data_path, device=device)
    train_sample, valid_sample, test_sample = get_train_test(
        data_path,
        fold_num=args.fold_num,
        condition=args.condition)

    scores = []
    for i in range(0, 1):
        print(f"============================ Fold {i} Start ============================")
        model_save_dir = os.path.join(args.data_path, args.kg_name+'+'+args.ddi_name, args.condition, f'Fold{i}')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        train_x_left = train_sample[i][:, 0]
        train_x_right = train_sample[i][:, 1]
        train_y = train_sample[i][:, 2:]

        valid_x_left = valid_sample[i][:, 0]
        valid_x_right = valid_sample[i][:, 1]
        valid_y = valid_sample[i][:, 2:]

        test_x_left = test_sample[i][:, 0]
        test_x_right = test_sample[i][:, 1]
        test_y = test_sample[i][:, 2:]

        train_y = torch.from_numpy(train_y).long()
        valid_y = torch.from_numpy(valid_y).long()
        test_y = torch.from_numpy(test_y).long()

        class_num = 65 if args.ddi_name == 'Deng' else 86
        model = CHIME(kg_g, smiles, args.hidden_dim, args.num_layer, class_num).to(device)
        loss_func = nn.CrossEntropyLoss()

        if i == 0:
            print(model)
        weight_p, bias_bn_emb_p = [], []
        for name, p in model.named_parameters():
            if 'bias' in name or 'bn' in name or 'embedding' in name:
                bias_bn_emb_p += [p]
            else:
                weight_p += [p]
        model_parameters = [
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_bn_emb_p, 'weight_decay': 0},
        ]

        optimizer = optim.AdamW(model_parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)

        best_val_score = None
        for epoch in range(args.epoch):
            train_one_epoch(model, loss_func, optimizer, train_x_left, train_x_right, train_y,
                            i, epoch, args.batch_size, device)
            
            val_score, _ = test(model, loss_func, valid_x_left, valid_x_right, valid_y, 
                                i, epoch, args.batch_size, device)

            val_f1 = val_score[1]
            if epoch > 40:
                early_stopping(val_f1, model)
                if early_stopping.counter == 0:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}.pth'))
                if early_stopping.early_stop or epoch == args.epoch - 1:
                    break
            if best_val_score is None:
                print(best_val_score)
            else:
                val_acc, val_f1, val_precision, val_recall, val_kappa = best_val_score
                print(
                    "val acc:{:.5f}, val f1:{:.5f}, val precision:{:.5f}, val recall:{:.5f}, val kappa:{:.5f}".format(
                        val_acc, val_f1, val_precision, val_recall, val_kappa
                    ))

        print(f"\n========== Fold {i} Training Finished. Evaluating on Test Set. ==========")
        best_model_path = os.path.join(model_save_dir, f'best_model_{args.num_layer}_{args.hidden_dim}.pth')
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        final_test_score = test(model, loss_func, test_x_left, test_x_right, test_y,
                                          i, 0, args.batch_size, args.label_type, device)

        scores.append(final_test_score)
        print('Test set score:', scores)

    scores = np.array(scores)
    mean_scores = scores.mean(axis=0)
    std_scores = scores.std(axis=0)

    print(f"\033[1;31mFinal DDI Result ({args.condition}):\n"
            f"Acc: {mean_scores[0]:.4f} ± {std_scores[0]:.4f}\n"
            f"F1 : {mean_scores[1]:.4f} ± {std_scores[1]:.4f}\n"
            f"Pre: {mean_scores[2]:.4f} ± {std_scores[2]:.4f}\n"
            f"Rec: {mean_scores[3]:.4f} ± {std_scores[3]:.4f}\033[0m")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='')
    ap.add_argument('--batch_size', type=int, default=2 ** 11)
    ap.add_argument('--fold_num', type=int, default=5)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layer', type=int, default=3)
    ap.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=40)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)

    ap.add_argument('--label_type', type=str, default='multi_class')
    ap.add_argument('--condition', type=str, choices=['S1', 'S2', 'S3'], default='S1')
    ap.add_argument('--data_path', type=str, default='./data')

    ap.add_argument('--kg_name', type=str, default='DRKG')
    ap.add_argument('--ddi_name', type=str, choices=['Deng', "Ryu"], default='Deng')

    args = ap.parse_args(args=[])
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    run(args)

