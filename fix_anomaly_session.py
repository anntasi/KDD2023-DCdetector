import numpy as np
import os

root = "expdata"

normal_fp = os.path.join(root, "test_normal", "session_test_normal.npy")
anom_fp = os.path.join(root, "test_anomaly", "session_test_anomaly.npy")

normal = np.load(normal_fp).astype(np.float32)
anom = np.load(anom_fp).astype(np.float32)

print("before:")
print("normal shape =", normal.shape)
print("anomaly shape =", anom.shape)

if anom.shape[0] < 10:
    prefix = normal[580:582]   # 2 packets
    suffix = normal[582:584]   # 2 packets
    anom = np.concatenate([prefix, anom, suffix], axis=0)

np.save(anom_fp, anom)

print("after:")
print("saved anomaly shape =", anom.shape)
print("saved to =", os.path.abspath(anom_fp))