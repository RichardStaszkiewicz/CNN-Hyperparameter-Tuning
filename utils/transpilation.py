from modules import plmodules as plm
import pytorch_lightning as pl
from datetime import timedelta
import numpy as np
import json


def count_mlp_in(config):
    """
    Counts the output size of the convolutional part in a ResNet-style CNN configuration.

    Parameters:
    - config (dict): CNN configuration dictionary with ResNet-style specifications.

    Returns:
    - int: Size of the output after the convolutional part.

    Notes:
    - The input `config` should be a dictionary containing ResNet-style CNN specifications.
    - The output size is calculated based on the convolutional layers specified in the configuration.
    - The calculation considers factors such as kernel size, padding, stride, and pooling.
    """
    if config["resnet_config"]["first_conv"]["padding"] == "same":
        width = np.ceil(
            config["init_width"] / config["resnet_config"]["first_conv"]["stride"]
        )
        length = np.ceil(
            config["init_length"] / config["resnet_config"]["first_conv"]["stride"]
        )
    else:
        width = np.floor(
            (
                config["init_width"]
                - config["resnet_config"]["first_conv"]["kernel_size"]
                + 2 * config["resnet_config"]["first_conv"]["padding"]
            )
            / config["resnet_config"]["first_conv"]["stride"]
            + 1
        )
        length = np.floor(
            (
                config["init_length"]
                - config["resnet_config"]["first_conv"]["kernel_size"]
                + 2 * config["resnet_config"]["first_conv"]["padding"]
            )
            / config["resnet_config"]["first_conv"]["stride"]
            + 1
        )

    for b in config["resnet_config"]["block_list"]:
        if b["padding"] == "same":
            width = np.ceil(width / b["stride"])
            length = np.ceil(length / b["stride"])
        else:
            width = np.floor(
                (width - b["kernel_size"] + 2 * b["padding"]) / b["stride"] + 1
            )
            length = np.floor(
                (length - b["kernel_size"] + 2 * b["padding"]) / b["stride"] + 1
            )

    width = np.floor(
        (width - config["resnet_config"]["pool_size"])
        / config["resnet_config"]["pool_size"]
        + 1
    )
    length = np.floor(
        (length - config["resnet_config"]["pool_size"])
        / config["resnet_config"]["pool_size"]
        + 1
    )
    return width * length * config["resnet_config"]["block_list"][-1]["out_channels"]


def actualise_config(config):
    """
    Repairs the configuration in case of inconsistent input and output sizes.

    Parameters:
    - config (dict): Configuration dictionary for a neural network.

    Returns:
    - dict: Repaired consistant configuration with rounded and adjusted values.

    Notes:
    - This function ensures that the configuration has consistent input and output sizes for each layer.
    - It rounds numerical values, adjusts configurations for ResNet blocks, and calculates input size for MLP.
    - The repaired configuration is returned.
    """
    config["batch_size"] = int(np.round(config["batch_size"]))
    resnet = [
        int(k.replace("resnet_out_channels_l", ""))
        for k in config.keys()
        if "resnet_out_channels_l" in k
    ]
    for l in resnet:
        config[f"resnet_out_channels_l{l}"] = int(
            np.round(config[f"resnet_out_channels_l{l}"])
        )
        config["resnet_config"]["block_list"][l]["out_channels"] = config[
            f"resnet_out_channels_l{l}"
        ]
        try:
            config["resnet_config"]["block_list"][l + 1]["in_channels"] = config[
                f"resnet_out_channels_l{l}"
            ]
        except Exception:
            pass
    resnet = [
        int(k.replace("resnet_kernel_size_l", ""))
        for k in config.keys()
        if "resnet_kernel_size_l" in k
    ]
    for l in resnet:
        config[f"resnet_kernel_size_l{l}"] = int(
            np.round(config[f"resnet_kernel_size_l{l}"])
        )
        config["resnet_config"]["block_list"][l]["kernel_size"] = config[
            f"resnet_kernel_size_l{l}"
        ]
    resnet = [
        int(k.replace("resnet_stride_l", ""))
        for k in config.keys()
        if "resnet_stride_l" in k
    ]
    for l in resnet:
        config[f"resnet_stride_l{l}"] = int(np.round(config[f"resnet_stride_l{l}"]))
        config["resnet_config"]["block_list"][l]["stride"] = config[
            f"resnet_stride_l{l}"
        ]
    resnet = [
        int(k.replace("resnet_padding_l", ""))
        for k in config.keys()
        if "resnet_padding_l" in k
    ]
    for l in resnet:
        config[f"resnet_padding_l{l}"] = int(np.round(config[f"resnet_padding_l{l}"]))
        config["resnet_config"]["block_list"][l]["padding"] = config[
            f"resnet_padding_l{l}"
        ]

    config["mlp_config"]["block_list"][0]["in_size"] = int(count_mlp_in(config))

    mlp = [int(k.replace("mlp_out_l", "")) for k in config.keys() if "mlp_out_l" in k]
    for l in mlp:
        config[f"mlp_out_l{l}"] = int(np.round(config[f"mlp_out_l{l}"]))
        config["mlp_config"]["block_list"][l]["out_size"] = config[f"mlp_out_l{l}"]
        try:
            config["mlp_config"]["block_list"][l + 1]["in_size"] = config[
                f"mlp_out_l{l}"
            ]
        except Exception:
            pass
    mlp = [int(k.replace("mlp_af_l", "")) for k in config.keys() if "mlp_af_l" in k]
    for af in mlp:
        config["mlp_config"]["block_list"][af]["activation_fun"] = config[
            f"mlp_af_l{af}"
        ]
    mlp = [int(k.replace("mlp_bn_l", "")) for k in config.keys() if "mlp_bn_l" in k]
    for bn in mlp:
        config[f"mlp_bn_l{bn}"] = int(np.round(config[f"mlp_bn_l{bn}"]))
        config["mlp_config"]["block_list"][bn]["batch_norm"] = config[f"mlp_bn_l{bn}"]
    mlp = [int(k.replace("mlp_do_l", "")) for k in config.keys() if "mlp_do_l" in k]
    for do in mlp:
        config["mlp_config"]["block_list"][do]["dropout"] = config[f"mlp_do_l{do}"]
    return config


def run_with_tune_ray(config, epochs=50, callbacks=[]):
    """
    Runs training using PyTorch Lightning with Ray Tune integration.

    Parameters:
    - config (dict): Configuration dictionary for the neural network.
    - epochs (int, optional): Number of training epochs (default is 50).
    - callbacks (list, optional): List of PyTorch Lightning callbacks (default is an empty list).

    Notes:
    - This function prepares and runs training using PyTorch Lightning with Ray Tune integration.
    - It first repairs the configuration using the `actualise_config` function.
    - A PyTorch Lightning model (`MNISTClassifier`) and data module (`MNISTDataModule`) are instantiated.
    - The training is performed using a PyTorch Lightning Trainer with specified epochs, callbacks, and settings.
    """
    config = actualise_config(config)
    model = plm.MNISTClassifier(config)
    dm = plm.MNISTDataModule(config["batch_size"])
    trainer = pl.Trainer(
        max_epochs=epochs,
        fast_dev_run=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)


def run_with_tune(config, max_time=60, epochs=50):
    """
    Runs training and validation using PyTorch Lightning with Ray Tune integration.

    Parameters:
    - config (dict): Configuration dictionary for the neural network.
    - max_time (int, optional): Maximum time allowed for training in seconds (default is 60).
    - epochs (int, optional): Number of training epochs (default is 50).

    Returns:
    - list: List of results from the validation phase.

    Notes:
    - This function prepares and runs training using PyTorch Lightning with Ray Tune integration.
    - It first repairs the configuration using the `actualise_config` function.
    - A PyTorch Lightning model (`MNISTClassifier`) and data module (`MNISTDataModule`) are instantiated.
    - The training is performed using a PyTorch Lightning Trainer with specified epochs, max time, and settings.
    - Results from the validation phase are returned, and an exception message is printed in case of failure.
    """
    config = actualise_config(config)
    # print(json.dumps(config, indent=4))
    model = plm.MNISTClassifier(config)
    dm = plm.MNISTDataModule(config["batch_size"])
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=epochs,
        max_time=timedelta(seconds=max_time),
        enable_checkpointing=False,
        fast_dev_run=False,
        callbacks=[],
    )
    try:
        trainer.fit(model, dm)
        results = trainer.validate(model, dm)
        return results
    except Exception as e:
        print(e)
        return [{"ptl/val_loss": np.Inf}]


def fn(x, HP_TUNED, METRIC, default_config, max_time_trial=60, debug=False):
    """
    Objective function for hyperparameter optimization using Ray Tune.

    Parameters:
    - x (list): Population as a vector of hyperparameters.
    - HP_TUNED (list): List of hyperparameters to tune.
    - METRIC (str): Evaluation metric to optimize.
    - default_config (dict): Default configuration dictionary for the neural network.
    - max_time_trial (int, optional): Maximum time allowed for training in seconds for each trial (default is 60).
    - debug (bool, optional): Flag for printing debug information (default is False).

    Returns:
    - float: Value of the specified evaluation metric for the given hyperparameter configuration.

    Notes:
    - This function serves as the objective function for hyperparameter optimization using Ray Tune.
    - It takes a population vector 'x' and updates the corresponding hyperparameters in the configuration.
    - The configuration is then used to run training with `run_with_tune` function, and the specified metric is extracted.
    - The value of the evaluation metric is returned for optimization.
    """
    # x - population as vector of hp
    config = default_config.copy()
    if debug:
        print(json.dumps(dict(zip(HP_TUNED, x)), indent=4))
    config.update(dict(zip(HP_TUNED, x)))
    results = run_with_tune(config, max_time=max_time_trial)
    return results[0][METRIC]


def fx(HP_TUNED, METRIC, default_config, max_time_tral=60, debug=False):
    """
    Returns a lambda function for hyperparameter optimization black-box evaluation function using Ray Tune.

    Parameters:
    - HP_TUNED (list): List of hyperparameters to tune.
    - METRIC (str): Evaluation metric to optimize.
    - default_config (dict): Default configuration dictionary for the neural network.
    - max_time_trial (int, optional): Maximum time allowed for training in seconds for each trial (default is 60).
    - debug (bool, optional): Flag for printing debug information (default is False).

    Returns:
    - callable: Lambda function that takes a population vector and returns the value of the specified metric.

    Notes:
    - This function returns a lambda function that can be used as the objective function
      for hyperparameter optimization using Ray Tune.
    - The lambda function is created based on the parameters passed to the `fn` function.
    - The returned lambda function takes a population vector 'x' and returns the value of the specified metric.
    """
    return lambda x: fn(x, HP_TUNED, METRIC, default_config, max_time_tral, debug)
