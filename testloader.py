from data_factory.data_loader import get_loader_segment

loader = get_loader_segment(
    index=0,
    data_path="./expdata",
    batch_size=4,
    win_size=10,
    step=1,
    mode="train",
    dataset="TNS"
)

batch = next(iter(loader))

print("len(batch) =", len(batch))
print("x shape =", batch[0].shape)
print("y shape =", batch[1].shape)
print("pkt_idx shape =", batch[2].shape)