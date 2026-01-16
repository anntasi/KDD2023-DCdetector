import pandas as pd
import numpy as np
import pandas as pd

TARGET_CONN = "10.116.192.4:1527<->10.116.55.127:55054"

scores = np.load("result/TNS/pdu_scores.npy")
sess_df = pd.read_csv("/home/dbsecure/project/c2b4ee28.7a9623b0.2101_10.116.192.4_tns_dataset/plots_features/data/sessions_index.csv")

row = sess_df[sess_df["conn_id"] == TARGET_CONN].iloc[0]
indices = list(map(int, row["meta_row_indices"].split(",")))

sess_scores = scores[indices]

max_pos = int(sess_scores.argmax())
max_score = float(sess_scores[max_pos])
max_pdu_index = indices[max_pos]

print("conn_id:", TARGET_CONN)
print("num_pdus:", len(indices))
print("max_pdu_index:", max_pdu_index)
print("max_pdu_score:", max_score)

CENTER = 386
R = 5  # 看前後各 5 個 PDU

meta = pd.read_csv("/home/dbsecure/project/c2b4ee28.7a9623b0.2101_10.116.192.4_tns_dataset/plots_features/data/tns_meta.csv")

cols = [
    "tns_index",
    "epoch",
    "pkt_type",
    "length_tns",
    "direction",
    "delta_t_in_conn",
    "conn_id"
]

print(meta.loc[CENTER-R:CENTER+R, cols])

