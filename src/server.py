from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import argparse, json, os


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
        fs.write(
            json.dumps(
                {
                    "type": "aggregation",
                    "id": "server",
                    "acc": sum(accuracies) / sum(examples),
                }
            )
            + "\n"
        )
        os.fsync(fs.fileno())
    return {"accuracy": sum(accuracies) / sum(examples)}


# Parse inputs
parser = argparse.ArgumentParser(description="Launches FL clients.")
parser.add_argument(
    "-clients",
    "--clients",
    type=int,
    default=2,
    help="Define the number of clients to be part of he FL process",
)
parser.add_argument(
    "-min",
    "--min",
    type=int,
    default=2,
    help="Minimum number of available clients",
)
parser.add_argument(
    "-rounds",
    "--rounds",
    type=int,
    default=10,
    help="Number of FL rounds",
)

args = vars(parser.parse_args())
num_clients = args["clients"]
min_clients = args["min"]
rounds = args["rounds"]
with open("/var/log/app.log", encoding="ascii", mode="a") as fs:
    fs.write(
        json.dumps(
            {
                "type": "startup",
                "id": "server",
                "num of clients": num_clients,
                "rounds": rounds,
            }
        )
        + "\n"
    )
    os.fsync(fs.fileno())
# Define strategy
# strategy = fl.server.strategy.FedAvg(
#     evaluate_metrics_aggregation_fn=weighted_average,
#     min_fit_clients=num_clients,
#     min_available_clients=min_clients,
# )


strategy = fl.server.strategy.FedProx(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=num_clients,
    min_available_clients=min_clients,
    proximal_mu=0.001,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy,
)
