import argparse
from adamp import AdamP


def boolean_string(s: str) -> bool:
    """
    Checks whether the string represents a boolean value.

    Args:
        s (str): the string to check.

    Raises:
        ValueError: raises when s not in {"False", "True"}.

    Returns:
        bool: whether the string represents a boolean value.
        
    Author: Adam
    """
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def parse_sweep_args() -> argparse.Namespace:
    """
    Parse hyperparameters from json file.
    
    Returns:
        - argparse.Namespace: Namespace with attributes named after json keys
        
    Author: Adam
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--nesterov", type=boolean_string)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--channels", type=int)
    parser.add_argument("--out_channels", type=int)
    parser.add_argument("--n_blocks", type=int)
    parser.add_argument("--block_types", type=str)
    parser.add_argument("--dropout_pb", type=float)
    parser.add_argument("--number_of_ensemble_models", type=int)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--entity_name", type=str)
    parser.add_argument('--normalize', type=bool)
    parser.add_argument('--size', type=int)
    return parser.parse_args()


def load_hyperparams():
    """
    Loads the hyperparameters from the parser to the dictionary.

    Returns:
        Dict[str, Any]: a dictionary of hyperparameters.
        
    Author: Adam
    """
    parser = parse_sweep_args()
    hyperparams = {}
    hyperparams["learning_rate"] = parser.learning_rate
    hyperparams["nesterov"] = parser.nesterov
    hyperparams["weight_decay"] = parser.weight_decay
    hyperparams["batch_size"] = parser.batch_size
    hyperparams["label_smoothing"] = parser.label_smoothing
    hyperparams["channels"] = parser.channels
    hyperparams["out_channels"] = parser.out_channels
    hyperparams["n_blocks"] = parser.n_blocks
    hyperparams["block_types"] = parser.block_types
    hyperparams["dropout_pb"] = parser.dropout_pb
    hyperparams["number_of_ensemble_models"] = parser.number_of_ensemble_models
    hyperparams["project_name"] = parser.project_name
    hyperparams["entity_name"] = parser.entity_name
    hyperparams["normalize"] = parser.normalize
    hyperparams["size"] = parser.size
    
    return hyperparams
