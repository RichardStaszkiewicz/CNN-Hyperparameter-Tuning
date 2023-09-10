from ray import tune
import numpy as np

GS_Search = {
    "lr": tune.grid_search(np.array(range(1, 100, 2)) / 1000),
}

RS_Search = {"lr": tune.uniform(0.001, 0.1)}
