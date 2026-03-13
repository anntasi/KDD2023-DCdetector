import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *

from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
#from metrics.metrics import *
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
            os.makedirs(path, exist_ok=True)

            ckpt_path = os.path.join(path, str(self.dataset) + '_checkpoint.pth')
            torch.save(model.state_dict(), ckpt_path)

            print("[INFO] Saved checkpoint to:", ckpt_path)

            self.val_loss_min = val_loss
            self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, 'dataset/'+self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.win_size = config.get('win_size', 10)
        self.input_c = config.get('input_c', 9)
        self.output_c = config.get('output_c', 9)

        self.n_heads = config.get('n_heads', 1)
        self.d_model = config.get('d_model', 256)
        self.e_layers = config.get('e_layers', 3)
        self.patch_size = config.get('patch_size', [5])

        self.lr = config.get('lr', 1e-4)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)

        self.dataset = config.get('dataset', 'TNS')
        self.data_path = config.get('data_path', '../expdata')
        self.model_save_path = config.get('model_save_path', 'checkpoints')
        self.anormly_ratio = config.get('anormly_ratio', 4.0)
        self.build_model()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def _aggregate_window_scores_to_packets(self, window_scores, window_pkt_idx, total_packets):
        """
        將 window-level scores 聚合成 packet-level scores。

        參數
        ----
        window_scores : np.ndarray, shape = (num_windows,)
            每個 window 的分數

        window_pkt_idx : np.ndarray, shape = (num_windows, win_size)
            每個 window 對應到哪些 packet index

        total_packets : int
            這個 loader / dataset 總共有多少 packet

        回傳
        ----
        packet_scores : np.ndarray, shape = (total_packets,)
            每個 packet 的平均分數

        valid_mask : np.ndarray, shape = (total_packets,)
            哪些 packet 至少被一個 window 覆蓋到
        """
        score_sum = np.zeros(total_packets, dtype=np.float64)
        score_count = np.zeros(total_packets, dtype=np.int64)

        for w in range(window_pkt_idx.shape[0]):
            s = float(window_scores[w])

            for t in range(window_pkt_idx.shape[1]):
                p = int(window_pkt_idx[w, t])

                if 0 <= p < total_packets:
                    score_sum[p] += s
                    score_count[p] += 1

        packet_scores = np.zeros(total_packets, dtype=np.float64)
        valid_mask = score_count > 0

        packet_scores[valid_mask] = score_sum[valid_mask] / score_count[valid_mask]

        return packet_scores, valid_mask    
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, batch in enumerate(vali_loader):
            if len(batch) == 3:
                input_data, _, _ = batch
            else:
                input_data, _ = batch
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                if len(batch) == 3:
                    input_data, labels, pkt_idx = batch
                else:
                    input_data, labels = batch
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss 

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
 
                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        
        for i, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                input_data, labels, pkt_idx = batch
            else:
                input_data, labels = batch    
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        thre_pkt_idx_all = []
        
        for i, batch in enumerate(self.thre_loader):
            if len(batch) == 3:
                input_data, labels, pkt_idx = batch
                thre_pkt_idx_all.append(pkt_idx.cpu().numpy())
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        thre_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, thre_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        #for i, batch in enumerate(self.thre_loader):
        test_pkt_idx_all = []
        for i, batch in enumerate(self.test_loader):
            if len(batch) == 3:
                input_data, labels, pkt_idx = batch
                test_pkt_idx_all.append(pkt_idx.cpu().numpy())
            else:
                input_data, labels = batch
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels.cpu().numpy())
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        matrix = [self.index]
        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))
        
        
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import accuracy_score

        
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        # =========================================================
        # Packet-level scoring
        # =========================================================
        thre_pkt_idx = np.concatenate(thre_pkt_idx_all, axis=0)
        test_pkt_idx = np.concatenate(test_pkt_idx_all, axis=0)

        thre_dataset = self.thre_loader.dataset
        test_dataset = self.test_loader.dataset

        thre_packet_scores, thre_valid_mask = self._aggregate_window_scores_to_packets(
            window_scores=thre_energy,
            window_pkt_idx=thre_pkt_idx,
            total_packets=len(thre_dataset.packet_labels_raw)
        )

        test_packet_scores, test_valid_mask = self._aggregate_window_scores_to_packets(
            window_scores=test_energy,
            window_pkt_idx=test_pkt_idx,
            total_packets=len(test_dataset.packet_labels_raw)
        )

        # 再跟 loader 內建的 selected_packet_mask 取交集
        thre_valid_mask = thre_valid_mask & thre_dataset.selected_packet_mask
        test_valid_mask = test_valid_mask & test_dataset.selected_packet_mask

        gt_packet = test_dataset.packet_labels_raw[test_valid_mask]

        threshold_packet = np.percentile(
            thre_packet_scores[thre_valid_mask],
            100 - self.anormly_ratio
        )

        pred_packet = (test_packet_scores[test_valid_mask] > threshold_packet).astype(int)

        accuracy_packet = accuracy_score(gt_packet, pred_packet)
        precision_packet, recall_packet, f_score_packet, _ = precision_recall_fscore_support(
            gt_packet,
            pred_packet,
            average='binary',
            zero_division=0
        )

        print("Packet Threshold : {:0.6f}".format(threshold_packet))
        print(
            "Packet Accuracy : {:0.4f}, Packet Precision : {:0.4f}, Packet Recall : {:0.4f}, Packet F-score : {:0.4f}".format(
                accuracy_packet, precision_packet, recall_packet, f_score_packet
            )
        )

        # optional: save packet-level outputs
        os.makedirs("packet_result", exist_ok=True)
        np.save("packet_result/dcd_packet_scores.npy", test_packet_scores)
        np.save("packet_result/dcd_packet_valid_mask.npy", test_valid_mask.astype(np.int64))
        np.save("packet_result/dcd_packet_gt.npy", test_dataset.packet_labels_raw.astype(np.int64))

        print("[INFO] Saved packet-level outputs to: packet_result/")
        
        if self.data_path == 'UCR' or 'UCR_AUG':
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy, precision, recall, f_score
