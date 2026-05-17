import flwr as fl
# ===============================
# SEND ROUND INFO TO CLIENT
# ===============================
def fit_config(server_round: int):
    return {"server_round": server_round}

# ===============================
# METRIC AGGREGATION 
# ===============================
def weighted_avg(metrics):
    total_examples = sum([num_examples for num_examples, _ in metrics])

    acc = (
        sum([num_examples * m["accuracy"] for num_examples, m in metrics])
        / total_examples
    )
    recall = (
        sum([num_examples * m["recall"] for num_examples, m in metrics])
        / total_examples
    )

    return {"accuracy": acc, "recall": recall}

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2,
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=weighted_avg,
    evaluate_metrics_aggregation_fn=weighted_avg,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=12),
    strategy=strategy,
)
