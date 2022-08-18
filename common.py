import argparse
from adamp import AdamP


def boolean_string(s):
    """TODO."""
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_sweep_args() -> argparse.Namespace:
    """Parse hyperparameters from json file.
    Returns:
        - argparse.Namespace: Namespace with attributes named after json keys
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--fixres_max_epochs", type=int)
    parser.add_argument("--fixres_learning_rate", type=float)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--entity_name", type=str)
    parser.add_argument("--EMA", type=boolean_string)
    return parser.parse_args()


def load_hyperparams():
    """TODO."""
    parser = parse_sweep_args()
    hyperparams = {}
    hyperparams["learning_rate"] = parser.learning_rate
    hyperparams["weight_decay"] = parser.weight_decay
    hyperparams["batch_size"] = parser.batch_size
    hyperparams["label_smoothing"] = parser.label_smoothing
    hyperparams["fixres_max_epochs"] = parser.fixres_max_epochs
    hyperparams["fixres_learning_rate"] = parser.fixres_learning_rate
    hyperparams["project_name"] = parser.project_name
    hyperparams["entity_name"] = parser.entity_name
    hyperparams["EMA"] = parser.EMA
    
    return hyperparams


def get_optimizer(model, hyperparams):
    return AdamP(
        model.parameters(),
        lr=hyperparams["lr"],
        betas=(hyperparams["betas"][0], hyperparams["betas"][1]),
        weight_decay=hyperparams["weight_decay"],
        nesterov=hyperparams["nesterov"],
    )