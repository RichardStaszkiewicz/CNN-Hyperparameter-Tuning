from ray import tune

######### TRIAL 1 ###########
# Estimated space sample: 160
# Estimated Time: 2h
# Estimated dimensions: 7
#############################

RS_search_config = {
    "lr": tune.uniform(0.01, 0.1),
    "resnet_out_channels_l0": tune.randint(8, 25),
    "resnet_kernel_size_l0": tune.randint(2, 5),
    "resnet_stride_l0": tune.randint(1, 4),
    "mlp_out_l0": tune.randint(64, 129),
    "mlp_bn_l0": tune.choice([True, False]),
    "mlp_do_l0": tune.uniform(0.095, 0.305),
}

GS_search_config = {
    "lr": tune.grid_search([0.01, 0.1]),
    "resnet_out_channels_l0": tune.grid_search([8, 12, 16, 20, 24]),
    "resnet_kernel_size_l0": tune.grid_search([2, 3]),
    "resnet_stride_l0": tune.grid_search([1, 2]),
    "mlp_out_l0": tune.grid_search([64, 128]),
    "mlp_bn_l0": tune.grid_search([True, False]),
    "mlp_do_l0": tune.grid_search([0.1, 0.3]),
} # Trials: 2 * 5 * 2 * 2 * 2 * 2 = 160

DES_search_config = {
    "lr": (0.01, 0.1),
    "resnet_out_channels_l0": (8, 24),
    "resnet_kernel_size_l0": (2, 3),
    "resnet_stride_l0": (1, 2),
    "mlp_out_l0": (64, 128),
    "mlp_bn_l0": (0, 1),
    "mlp_do_l0": (0.095, 0.305),
}

DES_start_config = {
    "lr": 0.01,
    "resnet_out_channels_l0": 16,
    "resnet_kernel_size_l0": 3,
    "resnet_stride_l0": 2,
    "mlp_out_l0": 64,
    "mlp_bn_l0": 1,
    "mlp_do_l0": 0.1,
}
