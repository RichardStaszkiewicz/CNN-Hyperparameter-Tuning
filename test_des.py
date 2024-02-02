import torch
from utils import DES
from modules import plmodules as plm
import numpy as np
from datetime import timedelta
from utils import transpilation
import yaml
import json

fn = (
        lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    )

def test0():
    f = lambda x: x[0]**2
    par = [0.005]
    result = DES.des_classic(par, f)
    assert abs(result['value']) < 0.001

def test1():
    par = [-100, -100, -100, -100]
    kwargs = {
        "upper": np.array([-5, -23.3, 14, 11]),
        "lower": np.array([-101, -101, -101, -150]),
        "stopfitness": 1e-10,
        "lambda": 5,
        "time": 5,
        "diag": True,
    }
    result = DES.des_classic(par, fn, **kwargs)
    assert result['value'] < 570
    assert result['value'] >= 567.89
    assert (result['counts']['function'] >= 40000) or (result['time'] >= kwargs['time'])
    assert len(result['par']) == 4


def test2():
    result = DES.des_tuner_wrapper(
        evaluation_fc=fn,
        start_config={"x1": -50, "x2": -20, "x3": -100, "x4": 10},
        search_config={"x1": (-101, -5), "x2": (-101, -23.3), "x3": (-101, 14), "x4": (3, 11)},
    ).fit({"stopfitness": 1e-10, "lambda": 20, "time": 20, "diag": True})

    assert result['value'] < 577
    assert result['value'] >= 576.89
    assert (result['counts']['function'] >= 40000) or (result['time'] >= 20)
    assert len(result['par']) == 4

def test3():
    f = lambda x: x[0]**2
    par = [0.005]
    result = DES.des_classic(
        par, f, upper=0.1, lower=0.001, **{"lambda": 4, "stopfitness": 1, "budget": 15}
    )
    assert (result['counts']['function'] == 15) or (result['message'] == 'Stop fitness reached.')

def test4():
    with open("CNN-Hyperparameter-Tuning/model/configs/model.yaml", "r") as stream:
        default_config = yaml.safe_load(stream)
    default_config = default_config['model']
    result = DES.des_tuner_wrapper(
        transpilation.fx(["lr"], "ptl/val_loss", default_config["model"]),
        {"lr": 0.002, },
        {"lr": (0.001, 0.1)}
    ).fit(
        {
            "lambda": 4,
            "time": 360,
            "diag": 1
        }
    )

    assert result is not None
    assert result['time'] == 360

def test5():
    with open("CNN-Hyperparameter-Tuning/model/configs/model.yaml", "r") as stream:
        default_config = yaml.safe_load(stream)
    default_config = default_config['model']
    result = DES.des_tuner_wrapper(
        transpilation.fx(
            ["lr", "resnet_out_channels_l1"], "ptl/val_loss", default_config["model"]
        ),
        {"lr": 0.002, "resnet_out_channels_l1": 8},
        {"lr": (0.001, 0.1), "resnet_out_channels_l1": (4, 17)},
    ).fit({"lambda": 4, "time": 360, "diag": 1})

    assert result is not None
    assert result['time'] == 360

def test6():
    import model.search_spaces.trial1_search as ts
    with open("CNN-Hyperparameter-Tuning/model/configs/model.yaml", "r") as stream:
        default_config = yaml.safe_load(stream)
    default_config = default_config['model']
    des_tuner = DES.des_tuner_wrapper(
        evaluation_fc = transpilation.fx(
            HP_TUNED=np.array(list(ts.DES_start_config.keys())),
            METRIC="ptl/val_loss",
            default_config=default_config,
            max_time_tral=100,
            debug=True
        ),
        start_config = ts.DES_start_config,
        search_config = ts.DES_search_config
    )
    results = des_tuner.fit({"time": 200, "diag": 1})
    assert results is not None

if __name__ == "__main__":
    pass