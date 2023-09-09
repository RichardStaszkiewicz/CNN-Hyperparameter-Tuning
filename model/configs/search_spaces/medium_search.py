from ray import tune

RS_search_config = {
    "batch_size": tune.choice([64, 128, 256]),
    "lr": tune.uniform(0.01, 0.1),
    "mlp_out_l0": tune.uniform(64, 128),
    "mlp_af_l0": tune.choice(['relu', 'None']),
    "mlp_bn_l0": tune.choice([True, False]),
    "mlp_do_l0": tune.uniform(0.095, 0.305)
}

GS_search_config = {
    "batch_size": tune.grid_search([64, 128, 256]),
    "lr": tune.grid_search([0.01, 0.1]),
    "mlp_out_l0": tune.grid_search([64, 128]),
    "mlp_af_l0": tune.grid_search(['relu', 'None']),
    "mlp_bn_l0": tune.grid_search([True, False]),
    "mlp_do_l0": tune.grid_search([0.1, 0.3])
}