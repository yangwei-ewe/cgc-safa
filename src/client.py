import io, copy, torch, argparse, json, logging, time, warnings, os, sys
from io import StringIO
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1.  PyTorch  模型、train、test
# ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(_net: Net, trainloader: DataLoader, epochs=1, mu=0.001):
    """Train the model on the training set."""
    global_model = copy.deepcopy(_net)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(_net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            loss = criterion(_net(images.to(DEVICE)), labels.to(DEVICE))
            proximal_term = torch.tensor(0.0, device=DEVICE)
            for w, w_t in zip(_net.parameters(), global_model.parameters()):
                proximal_term += ((w - w_t.to(DEVICE)) ** 2).sum()
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()


def test(_net: Net, loader: DataLoader):
    _net.eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            outs = _net(imgs.to(DEVICE))
            lbls = lbls.to(DEVICE)
            loss += criterion(outs, lbls).item()
            total += lbls.size(0)
            correct += (outs.argmax(1) == lbls).sum().item()
    return loss / len(loader.dataset), correct / total


def load_npz_from_url(url: str, id: int) -> tuple[np.ndarray, np.ndarray]:
    response = requests.get(url + str(id))
    response.raise_for_status()

    npz_data = np.load(io.BytesIO(response.content))
    # print("datas: ", npz_data.files)
    return (npz_data["images"], npz_data["labels"])


def load_data(
    images: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8
) -> tuple[DataLoader, DataLoader]:

    # 如果 images 已經是 tensor，就可以跳過 ToTensor
    trf = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 轉成 tensor 並標準化
    tensor_images = torch.stack(
        [trf(torch.tensor(img, dtype=torch.float32)) for img in images]
    )
    tensor_labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(tensor_images, tensor_labels)

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(
        testset, batch_size=32, shuffle=False
    )


def wait_for_ready(url: str, max_tries=60, period=2):
    for i in range(max_tries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print("Distributor is ready.")
                return
        except requests.RequestException as e:
            print(f"Waiting for distributor... attempt {i+1}/{max_tries}", flush=True)
        time.sleep(period)
    raise TimeoutError(f"Distributor not ready after {max_tries * period} seconds.")


import requests


def get_cid(url) -> int:
    id = requests.get(url).text
    return id


parser = argparse.ArgumentParser(description="Launches FL clients.")
parser.add_argument(
    "-cid",
    "--cid",
    type=int,
    default=-1,
    help="Define Client_ID",
)
parser.add_argument(
    "-server",
    "--server",
    default="0.0.0.0",
    help="Server Address",
)
parser.add_argument(
    "-port",
    "--port",
    default="30051",
    help="Server Port",
)
parser.add_argument("-epoch", "--epoch", default=10, type=int, help="number of epochs")
parser.add_argument("-data", "--data", default="./data", help="Dataset source path")
args = vars(parser.parse_args())
cid = args["cid"]
if cid == -1:
    cid = get_cid("http://id-distributor-svc.fedprox.svc.cluster.local:30021/get")
server = args["server"]
port = args["port"]
datapath = args["data"]
epoch = args["epoch"]
net = Net().to(DEVICE)

max_tries = int(os.environ["max_tries"])
period = int(os.environ["period"])

print("get id:", int(cid), flush=True)
wait_for_ready("http://ds-distributor-svc:17500/ready/", max_tries, period)
images_data, labels_data = load_npz_from_url(
    "http://ds-distributor-svc.fedprox.svc.cluster.local:17500/ds/", cid
)
train_loader, test_loader = load_data(images_data, labels_data)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler(filename="/app/app.log")],
# )

SCALAR = bool | bytes | float | int | str


# ─────────────────────────────────────────────
# 3.  Flower Client
# ─────────────────────────────────────────────
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(
        self, config: dict[str, bool | bytes | float | int | str]
    ) -> np.ndarray:  # config unused
        print(config)
        return [p.detach().cpu().numpy() for p in net.state_dict().values()]

    def set_parameters(self, params):
        sd = OrderedDict(
            zip(net.state_dict().keys(), [torch.tensor(p) for p in params])
        )
        net.load_state_dict(sd, strict=True)

    def fit(
        self, parameters: np.ndarray, config: dict[str, SCALAR]
    ) -> tuple[np.ndarray, int, dict[str, SCALAR]]:
        self.set_parameters(parameters)
        with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
            fs.write(
                json.dumps(
                    {
                        "type": "start",
                        "id": cid,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                )
                + "\n"
            )
            os.fsync(fs.fileno())

        train(net, train_loader, epochs=epoch)
        with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
            fs.write(
                json.dumps(
                    {
                        "type": "end",
                        "id": cid,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                )
                + "\n"
            )
            os.fsync(fs.fileno())
        return self.get_parameters({}), len(train_loader.dataset), {}

    def evaluate(self, parameters: np.ndarray, config: dict[str, SCALAR]):
        self.set_parameters(parameters)
        loss, acc = test(net, test_loader)
        with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
            fs.write(
                json.dumps({"type": "eval", "id": cid, "loss": loss, "acc": acc}) + "\n"
            )
            os.fsync(fs.fileno())
        return loss, len(test_loader.dataset), {"accuracy": acc}


with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
    fs.write(json.dumps({"type": "startup", "id": cid, "epochs": epoch}) + "\n")
    os.fsync(fs.fileno())
# print(f"[CID {cid}] connecting to {server}:{port}")
fl.client.start_client(
    server_address=f"{server}:{port}",
    client=FlowerClient().to_client(),
)
