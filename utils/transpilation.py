from modules import plmodules as plm
import pytorch_lightning as pl
from datetime import timedelta

def actualise_config(config):
    mlp = [int(k.replace("mlp_out_l", "")) for k in config.keys() if "mlp_out_l" in k]
    for l in mlp:
        config["mlp_config"]["block_list"][l]["out_size"] = config[f"mlp_out_l{l}"]
        config["mlp_config"]["block_list"][l + 1]["in_size"] = config[f"mlp_out_l{l}"]
    mlp = [int(k.replace("mlp_af_l", "")) for k in config.keys() if "mlp_af_l" in k]
    for af in mlp:
        config["mlp_config"]["block_list"][af]["activation_fun"] = config[
            f"mlp_af_l{af}"
        ]
    mlp = [int(k.replace("mlp_bn_l", "")) for k in config.keys() if "mlp_bn_l" in k]
    for bn in mlp:
        config["mlp_config"]["block_list"][bn]["batch_norm"] = config[f"mlp_bn_l{bn}"]
    mlp = [int(k.replace("mlp_do_l", "")) for k in config.keys() if "mlp_do_l" in k]
    for do in mlp:
        config["mlp_config"]["block_list"][do]["dropout"] = config[f"mlp_do_l{do}"]
    return config

def run_with_tune(config, epochs=50):
    config = actualise_config(config)
    model = plm.MNISTClassifier(config)
    dm = plm.MNISTDataModule(config["batch_size"])
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=epochs,
        max_time=timedelta(seconds=60),
        fast_dev_run=False,
        callbacks=[],
    )
    trainer.fit(model, dm)
    results = trainer.validate(model, dm)
    return results


def fn(x, HP_TUNED, METRIC, default_config):
    # x - population as vector of hp
    config = default_config.copy()
    config.update(dict(zip(HP_TUNED, x)))
    results = run_with_tune(config)
    return results[0][METRIC]


def fx(HP_TUNED, METRIC, default_config):
    return lambda x: fn(x, HP_TUNED, METRIC, default_config)
