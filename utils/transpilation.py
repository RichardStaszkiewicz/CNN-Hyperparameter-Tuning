from modules import plmodules as plm
import pytorch_lightning as pl
from datetime import timedelta
import numpy as np
import json


def count_mlp_in(config):
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
    config = actualise_config(config)
    print(config)
    model = plm.MNISTClassifier(config)
    dm = plm.MNISTDataModule(config['batch_size'])
    trainer = pl.Trainer(
        max_epochs=epochs,
        fast_dev_run=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    trainer.fit(model, dm)

def run_with_tune(config, max_time=60, epochs=50):
    config = actualise_config(config)
    print(json.dumps(config, indent=4))
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
        # raise e
        print(e)
        return [{"ptl/val_loss": np.Inf}] # to be resolved


def fn(x, HP_TUNED, METRIC, default_config, max_time_trial=60, debug=False):
    # x - population as vector of hp
    config = default_config.copy()
    if debug:
        print(json.dumps(dict(zip(HP_TUNED, x)), indent=4))
    config.update(dict(zip(HP_TUNED, x)))
    results = run_with_tune(config, max_time=max_time_trial)
    return results[0][METRIC]


def fx(HP_TUNED, METRIC, default_config, max_time_tral=60, debug=False):
    return lambda x: fn(x, HP_TUNED, METRIC, default_config, max_time_tral, debug)
