from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
import json
from datetime import datetime
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

            ckpt_path = os.path.join(path, 'checkpoint.pth')
            torch.save(model.state_dict(), ckpt_path)

            print("[INFO] Saved checkpoint to:", ckpt_path)

            self.val_loss_min = val_loss
            self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {
        'loss_fuc': 'MSE',
        'model_save_path': 'checkpoints',
    }

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.win_size = config.get('win_size', 10)
        self.input_c = config.get('input_c', 9)
        self.output_c = config.get('output_c', 9)

        self.n_heads = config.get('n_heads', 1)
        self.d_model = config.get('d_model', 256)
        self.e_layers = config.get('e_layers', 3)
        self.patch_size = config.get('patch_size', [5])
        self.packet_score_mode = config.get("packet_score_mode", "position")
        self.lr = config.get('lr', 1e-4)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)

        self.dataset = config.get('dataset', 'TNS')
        self.data_path = config.get('data_path', '../expdata')
        self.anormly_ratio = config.get('anormly_ratio', 4.0)
        self.build_model()
        self._init_output_dirs()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
    
    def _pretty_print_packet_windows(
        self,
        position_scores,
        window_pkt_idx,
        packet_idx_list,
        packet_threshold=None,
        save_txt=True,
        max_windows_per_packet=None,
    ):
        """
        漂亮列印：
        對每個指定 packet，找出所有包含它的 windows，
        並印出該 window 內每個 packet 的 index 與對應 score。

        Parameters
        ----------
        position_scores : np.ndarray, shape = (num_windows, win_size)
            每個 window 每個位置的分數（例如 test_pos_energy）

        window_pkt_idx : np.ndarray, shape = (num_windows, win_size)
            每個 window 對應到哪些 packet index

        packet_idx_list : array-like
            想檢查的 packet index 列表（例如 anomaly packet indices）

        packet_threshold : float or None
            若有提供，會一起印出方便比對

        save_txt : bool
            是否把輸出也存成 txt

        max_windows_per_packet : int or None
            若不是 None，則每個 packet 最多只印前幾個 windows
        """
        import os
        import numpy as np

        lines = []
        sep_major = "=" * 110
        sep_minor = "-" * 110

        def add(line=""):
            print(line, flush=True)
            lines.append(line)

        add("\n" + sep_major)
        add("Pretty Inspect: packet-in-window position scores")
        if packet_threshold is not None:
            add(f"packet_threshold = {packet_threshold:.6f}")
        add(sep_major)

        for pkt in packet_idx_list:
            hit_windows = np.where(np.any(window_pkt_idx == pkt, axis=1))[0]

            add(f"\n[Target packet {pkt}] covered by {len(hit_windows)} windows")

            if len(hit_windows) == 0:
                add("  -> no window covers this packet")
                add(sep_minor)
                continue

            # 依照該 packet 在各 window 對應位置的 score 由高到低排序
            packet_hits = []
            for w in hit_windows:
                pos = np.where(window_pkt_idx[w] == pkt)[0]
                if len(pos) == 0:
                    continue
                pos = int(pos[0])
                target_score = float(position_scores[w, pos])
                packet_hits.append((w, pos, target_score))

            packet_hits.sort(key=lambda x: x[2], reverse=True)

            if max_windows_per_packet is not None:
                packet_hits = packet_hits[:max_windows_per_packet]

            add(sep_minor)

            for rank, (w, pos, target_score) in enumerate(packet_hits, start=1):
                pkt_list = window_pkt_idx[w]
                score_list = position_scores[w]

                add(
                    f"window_rank={rank:02d} | window_idx={w:4d} | "
                    f"target_pos={pos} | target_score={target_score:10.6f}"
                )
                add(" pos | packet_idx | score       | mark ")
                add("-----+------------+-------------+------")

                for t in range(len(pkt_list)):
                    p = int(pkt_list[t])
                    s = float(score_list[t])

                    marks = []
                    if p == pkt:
                        marks.append("TARGET")
                    if packet_threshold is not None and s > packet_threshold:
                        marks.append(">thr")

                    mark_str = ",".join(marks) if marks else ""

                    add(f"{t:>4d} | {p:>10d} | {s:>11.6f} | {mark_str}")

                add(sep_minor)

        if save_txt:
            out_path = os.path.join(self.analysis_dir, "pretty_packet_window_scores.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            add(f"[INFO] Saved pretty print to: {out_path}")
    def _inspect_anomaly_related_windows(
        self,
        scores,
        window_pkt_idx,
        anomaly_packet_idx,
        score_type="window",
        save_csv=True,
        top_k=10,
    ):
        """
        檢查 anomaly packet 相關的分數資訊。

        Parameters
        ----------
        scores : np.ndarray
            - 若 score_type="window": shape = (num_windows,)
            每個 window 的 anomaly score
            - 若 score_type="packet": shape = (total_packets,)
            每個 packet 的 anomaly score

        window_pkt_idx : np.ndarray, shape = (num_windows, win_size)
            每個 window 對應到哪些 packet index

        anomaly_packet_idx : np.ndarray, shape = (num_anomaly_packets,)
            anomaly packet 的原始 index

        score_type : str
            "window" 或 "packet"

        save_csv : bool
            是否存成 csv

        top_k : int
            額外列出全體最高分的前幾個，方便比較
        """
        import pandas as pd
        import numpy as np
        import os

        if score_type not in ["window", "packet"]:
            raise ValueError(f"Unknown score_type: {score_type}")

        rows = []

        print(f"\n========== Inspect anomaly-related {score_type} scores ==========", flush=True)

        # =========================================================
        # WINDOW MODE
        # =========================================================
        if score_type == "window":
            window_scores = scores

            q50 = np.percentile(window_scores, 50)
            q90 = np.percentile(window_scores, 90)
            q95 = np.percentile(window_scores, 95)
            q99 = np.percentile(window_scores, 99)

            print(
                "Global window score percentiles: "
                f"p50={q50:.6f}, p90={q90:.6f}, p95={q95:.6f}, p99={q99:.6f}",
                flush=True
            )

            for pkt in anomaly_packet_idx:
                hit_mask = np.any(window_pkt_idx == pkt, axis=1)
                hit_windows = np.where(hit_mask)[0]

                print(f"\n[Anomaly packet {pkt}] covered by {len(hit_windows)} windows", flush=True)

                if len(hit_windows) == 0:
                    print("  -> no window covers this packet", flush=True)
                    continue

                local_scores = window_scores[hit_windows]

                print(
                    "  local window score stats: "
                    f"min={local_scores.min():.6f}, "
                    f"max={local_scores.max():.6f}, "
                    f"mean={local_scores.mean():.6f}, "
                    f"std={local_scores.std():.6f}",
                    flush=True
                )

                sorted_idx = hit_windows[np.argsort(local_scores)[::-1]]

                for rank, w_idx in enumerate(sorted_idx, start=1):
                    score = float(window_scores[w_idx])
                    pkt_list = window_pkt_idx[w_idx].tolist()
                    percentile = float((window_scores <= score).mean() * 100.0)

                    print(
                        f"    rank={rank:02d} "
                        f"window_idx={w_idx} "
                        f"score={score:.6f} "
                        f"global_pct~={percentile:.2f} "
                        f"packets={pkt_list}",
                        flush=True
                    )

                    rows.append({
                        "anomaly_packet_idx": int(pkt),
                        "window_idx": int(w_idx),
                        "window_score": score,
                        "global_percentile_approx": percentile,
                        "window_pkt_idx": " ".join(map(str, pkt_list)),
                        "rank_within_this_anomaly_packet": int(rank),
                    })

            top_idx = np.argsort(window_scores)[::-1][:top_k]
            top_rows = []

            print(f"\nTop-{top_k} highest-scoring windows globally:", flush=True)
            for rank, w_idx in enumerate(top_idx, start=1):
                score = float(window_scores[w_idx])
                pkt_list = window_pkt_idx[w_idx].tolist()
                print(
                    f"  global_rank={rank:02d} window_idx={w_idx} "
                    f"score={score:.6f} packets={pkt_list}",
                    flush=True
                )
                top_rows.append({
                    "global_rank": int(rank),
                    "window_idx": int(w_idx),
                    "window_score": score,
                    "window_pkt_idx": " ".join(map(str, pkt_list)),
                })

            if save_csv:
                detail_df = pd.DataFrame(rows)
                top_df = pd.DataFrame(top_rows)

                detail_path = os.path.join(self.analysis_dir, "anomaly_related_windows.csv")
                top_path = os.path.join(self.analysis_dir, f"top_{top_k}_global_windows.csv")

                detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
                top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

                print(f"\n[INFO] Saved anomaly window inspection to: {detail_path}", flush=True)
                print(f"[INFO] Saved top-{top_k} global windows to: {top_path}", flush=True)

        # =========================================================
        # PACKET MODE
        # =========================================================
        elif score_type == "packet":
            packet_scores = scores

            q50 = np.percentile(packet_scores, 50)
            q90 = np.percentile(packet_scores, 90)
            q95 = np.percentile(packet_scores, 95)
            q99 = np.percentile(packet_scores, 99)

            print(
                "Global packet score percentiles: "
                f"p50={q50:.6f}, p90={q90:.6f}, p95={q95:.6f}, p99={q99:.6f}",
                flush=True
            )

            for rank, pkt in enumerate(anomaly_packet_idx, start=1):
                score = float(packet_scores[pkt])
                percentile = float((packet_scores <= score).mean() * 100.0)

                hit_mask = np.any(window_pkt_idx == pkt, axis=1)
                hit_windows = np.where(hit_mask)[0]

                print(
                    f"\n[Anomaly packet {pkt}] "
                    f"packet_score={score:.6f} "
                    f"global_pct~={percentile:.2f} "
                    f"covered_by_windows={len(hit_windows)}",
                    flush=True
                )

                rows.append({
                    "rank_within_anomaly_packets": int(rank),
                    "anomaly_packet_idx": int(pkt),
                    "packet_score": score,
                    "global_percentile_approx": percentile,
                    "covered_by_windows": int(len(hit_windows)),
                })

            top_idx = np.argsort(packet_scores)[::-1][:top_k]
            top_rows = []

            print(f"\nTop-{top_k} highest-scoring packets globally:", flush=True)
            for rank, pkt in enumerate(top_idx, start=1):
                score = float(packet_scores[pkt])
                print(
                    f"  global_rank={rank:02d} packet_idx={pkt} "
                    f"score={score:.6f}",
                    flush=True
                )
                top_rows.append({
                    "global_rank": int(rank),
                    "packet_idx": int(pkt),
                    "packet_score": score,
                })

            if save_csv:
                detail_df = pd.DataFrame(rows)
                top_df = pd.DataFrame(top_rows)

                detail_path = os.path.join(self.analysis_dir, "anomaly_related_packets.csv")
                top_path = os.path.join(self.analysis_dir, f"top_{top_k}_global_packets.csv")

                detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
                top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

                print(f"\n[INFO] Saved anomaly packet inspection to: {detail_path}", flush=True)
                print(f"[INFO] Saved top-{top_k} global packets to: {top_path}", flush=True)
                
    def _init_output_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        patch_str = "-".join(map(str, self.patch_size)) if isinstance(self.patch_size, (list, tuple)) else str(self.patch_size)

        self.run_name = f"{timestamp}_ws{self.win_size}_ps{patch_str}_bs{self.batch_size}_ep{self.num_epochs}"
        self.output_root = os.path.join("outputs", self.dataset, self.run_name)

        self.checkpoint_dir = os.path.join("outputs", self.dataset, "shared_checkpoints")
        self.analysis_dir = os.path.join(self.output_root, "analysis")
        self.inference_dir = os.path.join(self.output_root, "inference")

        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.inference_dir, exist_ok=True)

        self.train_history = []

        config_to_save = {
            "dataset": self.dataset,
            "data_path": self.data_path,
            "index": self.index,
            "win_size": self.win_size,
            "input_c": self.input_c,
            "output_c": self.output_c,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "e_layers": self.e_layers,
            "patch_size": self.patch_size,
            "lr": self.lr,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "anormly_ratio": self.anormly_ratio,
            "model_save_path": self.model_save_path,
            "packet_score_mode": self.packet_score_mode,
        }

        with open(os.path.join(self.output_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)

        print("[INFO] Output directory:", self.output_root)
        
        
        
    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    def _aggregate_position_scores_to_packets(self, position_scores, window_pkt_idx, total_packets):
        """
        將 position-aware scores 聚合成 packet-level scores。

        參數
        ----
        position_scores : np.ndarray, shape = (num_windows, win_size)
            每個 window 內每個位置的分數

        window_pkt_idx : np.ndarray, shape = (num_windows, win_size)
            每個 window 對應到哪些 packet index

        total_packets : int
            dataset 總 packet 數

        回傳
        ----
        packet_scores : np.ndarray, shape = (total_packets,)
            每個 packet 的平均分數

        valid_mask : np.ndarray, shape = (total_packets,)
            哪些 packet 至少被一個 window 覆蓋到
        """
        score_sum = np.zeros(total_packets, dtype=np.float64)
        score_count = np.zeros(total_packets, dtype=np.int64)

        num_windows, win_size = window_pkt_idx.shape

        for w in range(num_windows):
            for t in range(win_size):
                p = int(window_pkt_idx[w, t])

                if 0 <= p < total_packets:
                    s = float(position_scores[w, t])
                    score_sum[p] += s
                    score_count[p] += 1

        packet_scores = np.zeros(total_packets, dtype=np.float64)
        valid_mask = score_count > 0
        packet_scores[valid_mask] = score_sum[valid_mask] / score_count[valid_mask]

        return packet_scores, valid_mask    
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

        return np.average(loss_1), 0.0
        #return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.checkpoint_dir
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

            vali_loss1, vali_loss2 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            
            epoch_record = {
                "epoch": epoch + 1,
                "epoch_time_sec": time.time() - epoch_time,
                "vali_loss1": float(vali_loss1),
                "vali_loss2": float(vali_loss2),
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.train_history.append(epoch_record)

            pd.DataFrame(self.train_history).to_csv(
                os.path.join(self.output_root, "train_history.csv"),
                index=False
            )
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        print("dataset window_labels shape =", self.test_loader.dataset.window_labels.shape, flush=True)
        print("dataset last window pkt_idx =", self.test_loader.dataset.window_packet_indices[-1], flush=True)#[-1] 是最後一個
        self.model.load_state_dict(
            torch.load(os.path.join(self.checkpoint_dir, 'checkpoint.pth')))#把之前 training 存的 model weights 載入回來
        self.model.eval()#把模型切到 evaluation mode。
        temperature = 50

        # (1) stastic on the train set:用 train set 蒐集一批「正常資料的 score」
        train_window_energy = []
        train_pkt_idx_all = []
        train_pos_energy = []
        for i, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                input_data, labels, pkt_idx = batch
                train_pkt_idx_all.append(pkt_idx.cpu().numpy())
            else:
                input_data, labels = batch    
            input = input_data.float().to(self.device)
            series, prior = self.model(input)#series：模型從資料學到的 attention / relation
            series_loss = 0.0                #prior：另一個對照的 attention / relation後面你是拿它們做 KL divergence，比較兩者差異。
            prior_loss = 0.0
            for u in range(len(prior)):#prior 和 series 應該不是單一 tensor，而是 多層 / 多個 attention map。所以你要對每一層都算 loss，再累加。
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

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # cri = metric.detach().cpu().numpy()
            cri = (-series_loss - prior_loss).detach().cpu().numpy()
            train_pos_energy.append(cri)
            
            cri_window = cri.max(axis=1)#該window分數是window內所有分數的最大值嗎
            train_window_energy.append(cri_window)
            

        train_energy = np.concatenate(train_window_energy, axis=0).reshape(-1)           # window-level
        train_pos_energy = np.concatenate(train_pos_energy, axis=0)       
        
        train_pkt_idx = np.concatenate(train_pkt_idx_all, axis=0)
        train_dataset = self.train_loader.dataset
        if self.packet_score_mode == "window":
            train_packet_scores, train_valid_mask = self._aggregate_window_scores_to_packets(
            window_scores=train_energy,
            window_pkt_idx=train_pkt_idx,
            total_packets=len(train_dataset.packet_labels_raw)
        )
        elif self.packet_score_mode == "position":
            train_packet_scores, train_valid_mask = self._aggregate_position_scores_to_packets(
                position_scores=train_pos_energy,
                window_pkt_idx=train_pkt_idx,
                total_packets=len(train_dataset.packet_labels_raw)
            )
            
        train_valid_mask = train_valid_mask & train_dataset.selected_packet_mask
        # (2) find the threshold
        thre_window_energy = []
        thre_pos_energy = []
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

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # cri = metric.detach().cpu().numpy()
            cri = (-series_loss - prior_loss).detach().cpu().numpy()
            cri_window = cri.max(axis=1)
            thre_window_energy.append(cri_window)
            thre_pos_energy.append(cri)

        thre_energy = np.concatenate(thre_window_energy, axis=0).reshape(-1)
        thre_pos_energy = np.concatenate(thre_pos_energy, axis=0)
        print(f"[INFO] packet_score_mode = {self.packet_score_mode}", flush=True)
        thre_pkt_idx = np.concatenate(thre_pkt_idx_all, axis=0)
        thre_dataset = self.thre_loader.dataset
        if self.packet_score_mode == "window":
            thre_packet_scores, thre_valid_mask = self._aggregate_window_scores_to_packets(
            window_scores=thre_energy,
            window_pkt_idx=thre_pkt_idx,
            total_packets=len(thre_dataset.packet_labels_raw)
        )

            combined_energy = np.concatenate(
                [train_energy, thre_energy],
                axis=0
            )

        elif self.packet_score_mode == "position":
            thre_packet_scores, thre_valid_mask = self._aggregate_position_scores_to_packets(
            position_scores=thre_pos_energy,
            window_pkt_idx=thre_pkt_idx,
            total_packets=len(thre_dataset.packet_labels_raw)
        )
            combined_energy = np.concatenate(
                [
                    train_packet_scores[train_valid_mask],
                    thre_packet_scores[thre_valid_mask]
                ],
                axis=0
            )

        else:
            raise ValueError(f"Unknown packet_score_mode: {self.packet_score_mode}")
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)#只有分數最高的 self.anormly_ratio% 會被當成 anomaly
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_window_energy = []
        test_pos_energy = []
        test_pkt_idx_all = []
       
      
        for i, batch in enumerate(self.test_loader):#一個batch有batch size個window
            if len(batch) == 3:
                input_data, labels, pkt_idx = batch
                print(f"[DEBUG][test_loop] batch={i} input_shape={input_data.shape} labels_shape={labels.shape}", flush=True)
                print(f"[DEBUG][test_loop] labels sum in batch = {labels.sum().item()}", flush=True)
                print(f"[DEBUG][test_loop] pkt_idx first={pkt_idx[0].cpu().numpy()} last={pkt_idx[-1].cpu().numpy()}", flush=True)
                test_pkt_idx_all.append(pkt_idx.cpu().numpy())
            else:
                input_data, labels = batch
                print(f"[DEBUG][test_loop] batch={i} input_shape={input_data.shape} labels_shape={labels.shape}", flush=True)
                print(f"[DEBUG][test_loop] labels sum in batch = {labels.sum().item()}", flush=True)
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
            # softmax       
            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # cri = metric.detach().cpu().numpy()
            
            #無softmax
            cri = (-series_loss - prior_loss).detach().cpu().numpy()
            print("cri.shape before agg =", cri.shape, flush=True)

            cri_window = cri.max(axis=1)   # 每個 window 聚合成一個 score
            print("cri_window.shape =", cri_window.shape, flush=True)
            test_window_energy.append(cri_window)
            test_pos_energy.append(cri)
            
            #windoe level用得
            # window_labels = labels.max(dim=1)[0]
            # test_labels.append(window_labels.cpu().numpy())
            
        test_energy = np.concatenate(test_window_energy, axis=0).reshape(-1)   # window-level
        test_pos_energy = np.concatenate(test_pos_energy, axis=0)               # position-level
        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)#每個 test window 的 true label
        # print("test_labels shape =", test_labels.shape, flush=True)
        # print("test_labels sum =", test_labels.sum(), flush=True)
        # print("test_labels tail 30 =", test_labels[-30:], flush=True)
        
        # test_labels = np.array(test_labels)
        print("len(test_energy) =", len(test_energy), flush=True)
        # print("len(test_labels) =", len(test_labels), flush=True)
        print("test_energy shape =", np.array(test_energy).shape, flush=True)
        # print("test_labels shape =", np.array(test_labels).shape, flush=True)
        # =========================================================
        # Window-level score analysis
        # =========================================================
        # normal_window_scores = test_energy[test_labels == 0]
        # anomaly_window_scores = test_energy[test_labels == 1]

        # print("num normal windows =", len(normal_window_scores), flush=True)
        # print("num anomaly windows =", len(anomaly_window_scores), flush=True)

        # if len(normal_window_scores) > 0:
        #     print("normal window score: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
        #         normal_window_scores.min(),
        #         normal_window_scores.max(),
        #         normal_window_scores.mean(),
        #         normal_window_scores.std()
        #     ), flush=True)

        # if len(anomaly_window_scores) > 0:
        #     print("anomaly window score: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
        #         anomaly_window_scores.min(),
        #         anomaly_window_scores.max(),
        #         anomaly_window_scores.mean(),
        #         anomaly_window_scores.std()
        #     ), flush=True)

        

        # plt.figure(figsize=(8, 5))
        # plt.hist(normal_window_scores, bins=50, alpha=0.5, label="normal window")
        # plt.hist(anomaly_window_scores, bins=50, alpha=0.5, label="anomaly window")
        # plt.axvline(thresh, linestyle='--', label=f'window threshold={thresh:.4f}')
        # plt.xlabel("window anomaly score")
        # plt.ylabel("count")
        # plt.title("Window Score Distribution")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.analysis_dir, "window_score_distribution.png"), dpi=200)
        # plt.close()

        # window_df = pd.DataFrame({
        #     "score": test_energy,
        #     "label": test_labels
        # })
        # window_df.to_csv(os.path.join(self.analysis_dir, "window_scores.csv"), index=False)
        # print(f"[INFO] Saved window score analysis to {self.analysis_dir}", flush=True)

        



        # pred = (test_energy > thresh).astype(int)#score > threshold → anomaly (1);否則 → normal (0)
        # gt = test_labels.astype(int)
        
        # matrix = [self.index]
        
        # window_pred_df = pd.DataFrame({
        #     "window_idx": np.arange(len(test_energy)),
        #     "window_score": test_energy,
        #     "window_label": gt,
        #     "window_pred": pred,
        # })
        # window_pred_df.to_csv(
        #     os.path.join(self.inference_dir, "window_predictions.csv"),
        #     index=False
        # )
        # scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        # for key, value in scores_simple.items():
        #     matrix.append(value)
        #     print('{0:21} : {1:0.4f}'.format(key, value))
        
        
        
        # 這是在做「point adjustment」
        # 這是 anomaly detection 論文裡很常見的後處理。
        # 概念是：
        # 如果某一段連續 ground truth anomaly 區間裡
        # 模型只抓到其中一個點
        # 那就把整段 anomaly 區間都算成有抓到
        # anomaly_state = False
        # for i in range(len(gt)):
        #     if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
        #         anomaly_state = True
        #         for j in range(i, 0, -1):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #         for j in range(i, len(gt)):
        #             if gt[j] == 0:
        #                 break
        #             else:
        #                 if pred[j] == 0:
        #                     pred[j] = 1
        #     elif gt[i] == 0:
        #         anomaly_state = False
        #     if anomaly_state:
        #         pred[i] = 1

        # pred = np.array(pred)
        # gt = np.array(gt)

        # from sklearn.metrics import precision_recall_fscore_support
        # from sklearn.metrics import accuracy_score

        # window_TP = np.sum((pred == 1) & (gt == 1))
        # window_FP = np.sum((pred == 1) & (gt == 0))
        # window_FN = np.sum((pred == 0) & (gt == 1))
        # window_TN = np.sum((pred == 0) & (gt == 0))
        # print("===========window level==============")
        # print("TP =", window_TP, flush=True)
        # print("FP =", window_FP, flush=True)
        # print("FN =", window_FN, flush=True)
        # print("TN =", window_TN, flush=True)
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        # =========================================================
        # Packet-level scoring
        # =========================================================
       
        test_pkt_idx = np.concatenate(test_pkt_idx_all, axis=0)

        
        test_dataset = self.test_loader.dataset

        print(f"[INFO] packet_score_mode = {self.packet_score_mode}", flush=True)

        if self.packet_score_mode == "window":
            
            test_packet_scores, test_valid_mask = self._aggregate_window_scores_to_packets(
                window_scores=test_energy,
                window_pkt_idx=test_pkt_idx,
                total_packets=len(test_dataset.packet_labels_raw)
            )

        elif self.packet_score_mode == "position":

            test_packet_scores, test_valid_mask = self._aggregate_position_scores_to_packets(
                position_scores=test_pos_energy,
                window_pkt_idx=test_pkt_idx,
                total_packets=len(test_dataset.packet_labels_raw)
            )

        else:
            raise ValueError(f"Unknown packet_score_mode: {self.packet_score_mode}")
            
        # 再跟 loader 內建的 selected_packet_mask 取交集
        thre_valid_mask = thre_valid_mask & thre_dataset.selected_packet_mask
        test_valid_mask = test_valid_mask & test_dataset.selected_packet_mask
        packet_threshold_source = np.concatenate([
            train_packet_scores[train_valid_mask],
            thre_packet_scores[thre_valid_mask]
        ], axis=0)
        threshold_packet = np.percentile(
            packet_threshold_source,
            100 - self.anormly_ratio
        )
        thresholds_dict = {
            "window_threshold": float(thresh),
            "packet_threshold": float(threshold_packet),
            "anormly_ratio": float(self.anormly_ratio),
        }

        with open(os.path.join(self.inference_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(thresholds_dict, f, indent=2, ensure_ascii=False)
        anom_idx_raw = np.where(test_dataset.packet_labels_raw == 1)[0]
        print("raw anomaly packet idx =", anom_idx_raw, flush=True)
        print("valid mask on anomaly idx =", test_valid_mask[anom_idx_raw], flush=True)
        
        # =========================================================
        # Inspect windows that cover anomaly packets
        # =========================================================
        

        # print("\n=== WINDOW VIEW ===", flush=True)
        # self._inspect_anomaly_related_windows(
        #     scores=test_energy,
        #     window_pkt_idx=test_pkt_idx,
        #     anomaly_packet_idx=anom_idx_raw,
        #     score_type="window",
        #     save_csv=True,
        #     top_k=10
        # )

        # print("\n=== PACKET VIEW ===", flush=True)
        # self._inspect_anomaly_related_windows(
        #     scores=test_packet_scores,
        #     window_pkt_idx=test_pkt_idx,
        #     anomaly_packet_idx=anom_idx_raw,
        #     score_type="packet",
        #     save_csv=True,
        #     top_k=10
        # )
        self._pretty_print_packet_windows(
            position_scores=test_pos_energy,
            window_pkt_idx=test_pkt_idx,
            packet_idx_list=anom_idx_raw,
            packet_threshold=threshold_packet,
            save_txt=True,
            max_windows_per_packet=None,   # 想限制就改成 5
        )
        selected_idx = np.where(test_dataset.selected_packet_mask)[0]
        print("selected packet idx head/tail =", selected_idx[:20], selected_idx[-20:], flush=True)
        
        gt_packet = test_dataset.packet_labels_raw[test_valid_mask]
        
        print("true anomaly after test_valid_mask =", gt_packet.sum(), flush=True)
        print("num anomaly removed by valid mask =",
      test_dataset.packet_labels_raw.sum() - gt_packet.sum(),
      flush=True)

        print("num_valid_test_packets =", test_valid_mask.sum(), flush=True)
        print("num_valid_thre_packets =", thre_valid_mask.sum(), flush=True)
        
        
        

        
        # =========================================================
        # Packet-level score analysis
        # =========================================================
        valid_test_packet_scores = test_packet_scores[test_valid_mask]
        normal_packet_scores = valid_test_packet_scores[gt_packet == 0]
        anomaly_packet_scores = valid_test_packet_scores[gt_packet == 1]
        print("=========packet level=============")
        print("num normal packets =", len(normal_packet_scores), flush=True)
        print("num anomaly packets =", len(anomaly_packet_scores), flush=True)

        if len(normal_packet_scores) > 0:
            print("normal packet score: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                normal_packet_scores.min(),
                normal_packet_scores.max(),
                normal_packet_scores.mean(),
                normal_packet_scores.std()
            ), flush=True)

        if len(anomaly_packet_scores) > 0:
            print("anomaly packet score: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                anomaly_packet_scores.min(),
                anomaly_packet_scores.max(),
                anomaly_packet_scores.mean(),
                anomaly_packet_scores.std()
            ), flush=True)

        plt.figure(figsize=(8, 5))
        plt.hist(normal_packet_scores, bins=50, alpha=0.5, label="normal packet")
        plt.hist(anomaly_packet_scores, bins=50, alpha=0.5, label="anomaly packet")
        plt.axvline(threshold_packet, linestyle='--', label=f'packet threshold={threshold_packet:.4f}')
        plt.xlabel("packet anomaly score")
        plt.ylabel("count")
        plt.title("Packet Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, "packet_score_distribution.png"), dpi=200)
        plt.close()

        packet_df = pd.DataFrame({
            "score": valid_test_packet_scores,
            "label": gt_packet
        })
        packet_df.to_csv(os.path.join(self.analysis_dir, "packet_scores_valid_only.csv"), index=False)
        
        print(f"[INFO] Saved packet score analysis to {self.analysis_dir}", flush=True)
        pred_packet = (test_packet_scores[test_valid_mask] > threshold_packet).astype(int)
        valid_packet_idx = np.where(test_valid_mask)[0]

        packet_pred_df = pd.DataFrame({
            "packet_idx": valid_packet_idx,
            "packet_score": test_packet_scores[test_valid_mask],
            "packet_label": gt_packet,
            "packet_pred": pred_packet,
        })
        packet_pred_df.to_csv(
            os.path.join(self.inference_dir, "packet_predictions.csv"),
            index=False
        )
        tp_indices = np.where(
            (pred_packet == 1) & (gt_packet == 1)
        )[0]

        print("TP packet indices:", tp_indices)
        packet_TP = np.sum((pred_packet == 1) & (gt_packet == 1))
        packet_FP = np.sum((pred_packet == 1) & (gt_packet == 0))
        packet_FN = np.sum((pred_packet == 0) & (gt_packet == 1))
        packet_TN = np.sum((pred_packet == 0) & (gt_packet == 0))
       
        print("TP =", packet_TP, flush=True)
        print("FP =", packet_FP, flush=True)
        print("FN =", packet_FN, flush=True)
        print("TN =", packet_TN, flush=True)

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
        
        np.save(os.path.join(self.inference_dir, "dcd_packet_scores.npy"), test_packet_scores)
        np.save(os.path.join(self.inference_dir, "dcd_packet_valid_mask.npy"), test_valid_mask.astype(np.int64))
        np.save(os.path.join(self.inference_dir, "dcd_packet_gt.npy"), test_dataset.packet_labels_raw.astype(np.int64))
        summary_metrics = {
            # "window_level": {
            #     "accuracy": float(accuracy),
            #     "precision": float(precision),
            #     "recall": float(recall),
            #     "f1": float(f_score),
            #     "TP": int(window_TP),
            #     "FP": int(window_FP),
            #     "FN": int(window_FN),
            #     "TN": int(window_TN),
            # },
            "packet_level": {
                "accuracy": float(accuracy_packet),
                "precision": float(precision_packet),
                "recall": float(recall_packet),
                "f1": float(f_score_packet),
                "TP": int(packet_TP),
                "FP": int(packet_FP),
                "FN": int(packet_FN),
                "TN": int(packet_TN),
            }
        }

        with open(os.path.join(self.output_root, "summary_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2, ensure_ascii=False)
            
        if self.data_path in ['UCR', 'UCR_AUG']:
            import csv
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

        return accuracy_packet, precision_packet, recall_packet, f_score_packet
