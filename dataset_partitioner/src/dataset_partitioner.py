import os
import argparse
import numpy as np
import requests
import tarfile
from collections import defaultdict
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from concurrent.futures import ThreadPoolExecutor


def dirichlet_split(cls_to_idxs, num_clients, alpha):
    clients = [[] for _ in range(num_clients)]
    for idxs in cls_to_idxs.values():
        proportions = np.random.dirichlet([alpha] * num_clients)
        idxs = np.random.permutation(idxs)
        split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        splits = np.split(idxs, split_points)
        for i, part in enumerate(splits):
            clients[i].extend(part.tolist())
    return clients


def download_and_extract_cifar10(data_dir):
    url = "http://nas.ican/datasets/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")

    print("Downloading CIFAR-10...", flush=True)
    response = requests.get(url, allow_redirects=True)
    with open(tar_path, "wb") as f:
        f.write(response.content)

    print("Extracting CIFAR-10...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)


def process_partition(client_id, indices, dataset, output_dir):
    images = np.stack([dataset[i][0].numpy() for i in indices], axis=0)
    labels = np.array([dataset[i][1] for i in indices], dtype=np.int64)
    out_path = os.path.join(output_dir, f"{client_id}.npz")
    np.savez(out_path, images=images, labels=labels)
    print(f"[Client {client_id}] Saved {len(indices)} samples.", flush=True)


def main():
    num_clients = int(os.environ.get("NUM_CLIENTS", 10))
    alpha = float(os.environ.get("ALPHA", 0.5))
    output_dir = os.environ.get("OUTPUT_DIR", "/data")
    data_dir = "./data"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Number of clients: {num_clients}")
    print(f"Alpha: {alpha}")
    print(f"Output directory: {output_dir}", flush=True)

    download_and_extract_cifar10(data_dir)

    print("Loading CIFAR-10 dataset...", flush=True)
    dataset = CIFAR10(root=data_dir, train=True, download=False, transform=ToTensor())

    cls_to_idxs = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        cls_to_idxs[label].append(idx)

    partitions = dirichlet_split(cls_to_idxs, num_clients, alpha)

    # 平行處理各 client 分割
    with ThreadPoolExecutor(max_workers=min(8, num_clients)) as executor:
        for client_id, indices in enumerate(partitions):
            executor.submit(process_partition, client_id, indices, dataset, output_dir)


if __name__ == "__main__":
    main()
