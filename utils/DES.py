import numpy as np
import random
import time


def des_classic(par, fn, lower=None, upper=None, **kwargs):
    print(kwargs)

    def control_param(name, default):
        v = kwargs[name] if name in kwargs.keys() else None
        return v if v is not None else default

    def sample_from_history(history, history_sample, lmbda):
        ret = []
        for _ in range(lmbda):
            ret.append(random.randint(0, len(history[history_sample[_]].T) - 1))
        return ret

    def delete_infs_nans(x):
        x[np.isnan(x)] = np.finfo(float).max
        x[np.isinf(x)] = np.finfo(float).max
        return x

    def fn_(x):
        if all(x >= lower) and all(x <= upper):
            nonlocal counteval
            counteval += 1
            return fn(x)
        else:
            return np.finfo(float).max

    def fn_l(P):
        if P.ndim > 1:
            if counteval + P.shape[1] <= budget:
                return np.apply_along_axis(fn_, 0, P)
            else:
                bud_left = budget - counteval
                ret = np.zeros(bud_left)
                if bud_left > 0:
                    for i in range(bud_left):
                        ret[i] = fn_(P[:, i])
                return np.concatenate(
                    (ret, np.repeat(np.finfo(float).max, P.shape[1] - bud_left))
                )
        else:
            if counteval < budget:
                return fn_(P)
            else:
                return np.finfo(float).max

    def fn_d(P, P_repaired, fitness):
        P = delete_infs_nans(P)
        P_repaired = delete_infs_nans(P_repaired)

        if P.ndim > 1 and P_repaired.ndim > 1:
            repaired_ind = np.all(P != P_repaired, axis=0)
            P_fit = fitness.copy()
            vec_dist = np.sum((P - P_repaired) ** 2, axis=0)
            P_fit[repaired_ind] = worst_fit + vec_dist[repaired_ind]
            P_fit = delete_infs_nans(P_fit)
            return P_fit
        else:
            P_fit = fitness.copy()
            if not np.array_equal(P, P_repaired):
                P_fit = worst_fit + np.sum((P - P_repaired) ** 2)
                P_fit = delete_infs_nans(P_fit)
            return P_fit

    def bounce_back_boundary2(x):
        if all(x >= lower) and all(x <= upper):
            return x
        elif any(x < lower):
            for i in np.where(x < lower)[0]:
                x[i] = lower[i] + abs(lower[i] - x[i]) % (upper[i] - lower[i])
        elif any(x > upper):
            for i in np.where(x > upper)[0]:
                x[i] = upper[i] - abs(upper[i] - x[i]) % (upper[i] - lower[i])
        x = delete_infs_nans(x)
        return bounce_back_boundary2(x)

    N = len(par)
    if lower is None:
        lower = np.full(N, -100)
    elif isinstance(lower, (int, float)):
        lower = np.full(N, lower)

    if upper is None:
        upper = np.full(N, 100)
    elif isinstance(upper, (int, float)):
        upper = np.full(N, upper)

    # Algorithm parameters
    Ft = control_param("Ft", 1)
    initFt = control_param("initFt", 1)
    stopfitness = control_param("stopfitness", -np.inf)
    budget = control_param("budget", 10000 * N)
    initlambda = control_param("lambda", 4 * N)
    lambda_ = initlambda
    mu = np.int64(control_param("mu", np.floor(lambda_ / 2)))
    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    weights_sum_s = np.sum(weights**2)
    mueff = control_param("mueff", (np.sum(weights) ** 2) / weights_sum_s)
    cc = control_param("ccum", mu / (mu + 2))
    path_length = control_param("pathLength", 6)
    cp = control_param("cp", 1 / np.sqrt(N))
    maxiter = control_param("maxit", np.floor(budget / (lambda_ + 1)))
    maxtime = control_param("time", np.Inf)
    c_Ft = control_param("c_Ft", 0)
    path_ratio = control_param("pathRatio", np.sqrt(path_length))
    hist_size = np.int64(control_param("history", np.ceil(6 + np.ceil(3 * np.sqrt(N)))))
    Ft_scale = ((mueff + 2) / (N + mueff + 3)) / (
        1
        + 2 * np.maximum(0, np.sqrt((mueff - 1) / (N + 1)) - 1)
        + (mueff + 2) / (N + mueff + 3)
    )
    tol = control_param("tol", 1e-12)
    counteval = 0
    sqrt_N = np.sqrt(N)

    log_all = control_param("diag", False)
    log_Ft = control_param("diag.Ft", log_all)
    log_value = control_param("diag.value", log_all)
    log_mean = control_param("diag.mean", log_all)
    log_mean_cord = control_param("diag.meanCords", log_all)
    log_pop = control_param("diag.pop", log_all)
    log_best_val = control_param("diag.bestVal", log_all)
    log_worst_val = control_param("diag.worstVal", log_all)
    log_eigen = control_param("diag.eigen", log_all)

    Lamarckism = control_param("Lamarckism", False)

    # Safety checks
    assert len(upper) == N
    assert len(lower) == N
    assert np.all(lower < upper)

    # Initialize variables
    best_fit = np.inf
    best_par = None
    worst_fit = None
    last_restart = 0
    restart_length = 0
    restart_number = 0

    # Preallocate logging structures
    if log_Ft:
        Ft_log = np.zeros((0, 1))
    if log_value:
        value_log = np.zeros((0, lambda_))
    if log_mean:
        mean_log = np.zeros((0, 1))
    if log_mean_cord:
        mean_cords_log = np.zeros((0, N))
    if log_pop:
        pop_log = np.zeros((N, lambda_, maxiter))
    if log_best_val:
        best_val_log = np.zeros((0, 1))
    if log_worst_val:
        worst_val_log = np.zeros((0, 1))
    if log_eigen:
        eigen_log = np.zeros((0, N))

    # Allocate buffers
    d_mean = np.zeros((N, hist_size))
    Ft_history = np.zeros(hist_size)
    pc = np.zeros((N, hist_size))

    # Initialize internal strategy parameters
    msg = None
    restart_number = -1
    time_start = time.time()

    while counteval < budget and ((time.time() - time_start) < maxtime):
        restart_number += 1
        mu = np.int64(np.floor(lambda_ / 2))
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        weights_pop = np.log(lambda_ + 1) - np.log(np.arange(1, lambda_ + 1))
        weights_pop /= np.sum(weights_pop)
        hist_head = 0
        iter_ = 0
        history = [None] * hist_size
        Ft = initFt

        # Create first population
        population = np.random.uniform(0.8 * lower, 0.8 * upper, size=(lambda_, N)).T

        cum_mean = (upper + lower) / 2
        population_repaired = np.apply_along_axis(bounce_back_boundary2, 0, population)

        if Lamarckism:
            population = population_repaired

        selection = np.zeros(mu, dtype=int)
        selected_points = np.zeros((N, mu))
        fitness = fn_l(population)
        old_mean = np.zeros(N)
        new_mean = np.copy(par)
        limit = 0
        worst_fit = np.max(fitness)

        # Store population and selection means
        pop_mean = np.matmul(population, weights_pop)
        mu_mean = np.copy(new_mean)

        # Matrices for creating diffs
        diffs = np.zeros((N, lambda_))
        x1_sample = np.zeros(lambda_, dtype=int)
        x2_sample = np.zeros(lambda_, dtype=int)

        chi_N = np.sqrt(N)
        hist_norm = 1 / np.sqrt(2)
        counter_repaired = 0
        stoptol = False

        while (
            counteval < budget
            and not stoptol
            and ((time.time() - time_start) < maxtime)
        ):
            iter_ += 1
            hist_head = (hist_head % hist_size) + 1
            mu = np.int64(np.floor(lambda_ / 2))
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)

            if log_Ft:
                Ft_log = np.vstack((Ft_log, Ft))
            if log_value:
                value_log = np.vstack((value_log, fitness))
            if log_mean:
                mean_log = np.vstack((mean_log, fn_l(bounce_back_boundary2(new_mean))))
            if log_mean_cord:
                mean_cords_log = np.vstack((mean_cords_log, new_mean))
            if log_pop:
                pop_log[:, :, iter_ - 1] = population
            if log_best_val:
                best_val_log = np.vstack(
                    (best_val_log, np.min([np.min(best_val_log), np.min(fitness)]))
                )
            if log_worst_val:
                worst_val_log = np.vstack(
                    (worst_val_log, np.max([np.max(worst_val_log), np.max(fitness)]))
                )
            if log_eigen:
                cov_matrix = np.cov(np.transpose(population))
                eigen_values = np.linalg.eigvals(cov_matrix)
                eigen_values = np.flip(np.sort(eigen_values))
                eigen_log = np.vstack((eigen_log, eigen_values))

            # Select best 'mu' individuals of population
            selection = np.argsort(fitness)[:mu]
            selected_points = population[:, selection]

            # Save selected population in the history buffer
            history[hist_head - 1] = selected_points * hist_norm / Ft

            # Calculate weighted mean of selected points
            old_mean = np.copy(new_mean)
            new_mean = np.matmul(selected_points, weights)

            # Write to buffers
            mu_mean = np.copy(new_mean)
            d_mean[:, hist_head - 1] = (mu_mean - pop_mean) / Ft

            step = (new_mean - old_mean) / Ft

            # Update Ft
            Ft_history[hist_head - 1] = Ft
            old_Ft = Ft

            # Update parameters
            if hist_head == 1:
                pc[:, hist_head - 1] = (1 - cp) * np.zeros(N) / np.sqrt(N) + np.sqrt(
                    mu * cp * (2 - cp)
                ) * step
            else:
                pc[:, hist_head - 1] = (1 - cp) * pc[:, hist_head - 2] + np.sqrt(
                    mu * cp * (2 - cp)
                ) * step

            # Sample from history with uniform distribution
            limit = hist_head if iter_ < hist_size else hist_size
            history_sample = np.random.choice(
                np.arange(0, limit), size=lambda_, replace=True
            )
            history_sample2 = np.random.choice(
                np.arange(0, limit), size=lambda_, replace=True
            )

            x1_sample = sample_from_history(history, history_sample, lambda_)
            x2_sample = sample_from_history(history, history_sample, lambda_)

            # Make diffs
            for i in range(lambda_):
                x1 = history[history_sample[i]][:, x1_sample[i]]
                x2 = history[history_sample[i]][:, x2_sample[i]]
                diffs[:, i] = (
                    np.sqrt(cc)
                    * (
                        (x1 - x2)
                        + np.random.randn(1) * d_mean[:, history_sample[i] - 1]
                    )
                    + np.sqrt(1 - cc)
                    * np.random.randn(1)
                    * pc[:, history_sample2[i] - 1]
                )

            # New population
            population = new_mean[:, np.newaxis] + Ft * diffs
            population += (
                tol
                * (max(1 - 2 / N**2, 0)) ** (iter_ / 2)
                * np.random.randn(*diffs.shape)
                / chi_N
            )
            population = delete_infs_nans(population)

            # Check constraints violations and repair the individual if necessary
            population_temp = population.copy()
            population_repaired = np.apply_along_axis(
                bounce_back_boundary2, 0, population
            )

            counter_repaired = np.sum(
                np.any(population_temp != population_repaired, axis=0)
            )

            if Lamarckism:
                population = population_repaired

            pop_mean = np.matmul(population, weights_pop)

            # Evaluation
            fitness = fn_l(population)
            if not Lamarckism:
                fitness_non_lamarckian = fn_d(population, population_repaired, fitness)

            # Break if fit
            wb = np.argmin(fitness)

            if fitness[wb] < best_fit:
                best_fit = fitness[wb]
                if Lamarckism:
                    best_par = population[:, wb]
                else:
                    best_par = population_repaired[:, wb]

            # Check worst fit
            ww = np.argmax(fitness)
            if fitness[ww] > worst_fit:
                worst_fit = fitness[ww]

            # Fitness with penalty for nonLamarckian approach
            if not Lamarckism:
                fitness = fitness_non_lamarckian

            # Check if the middle point is the best found so far
            cum_mean = 0.8 * cum_mean + 0.2 * new_mean
            cum_mean_repaired = bounce_back_boundary2(cum_mean)
            fn_cum = fn_l(cum_mean_repaired)

            if fn_cum < best_fit:
                best_fit = fn_cum
                best_par = cum_mean_repaired

            if fitness[0] <= stopfitness:
                msg = "Stop fitness reached."
                break

    exe_time = time.time() - time_start
    if exe_time > maxtime:
        msg = "Time limit reached"
        exe_time = maxtime

    cnt = {"function": int(counteval)}

    log = {}

    if log_Ft:
        log["Ft"] = Ft_log
    if log_value:
        log["value"] = value_log[:iter_, :]
    if log_mean:
        log["mean"] = mean_log[:iter_]
    if log_mean_cord:
        log["meanCord"] = mean_cords_log
    if log_pop:
        log["pop"] = pop_log[:, :, :iter_]
    if log_best_val:
        log["bestVal"] = best_val_log
    if log_worst_val:
        log["worstVal"] = worst_val_log
    if log_eigen:
        log["eigen"] = eigen_log

    res = {
        "par": best_par.tolist(),
        "value": best_fit,
        "counts": cnt,
        "resets": restart_number,
        "convergence": 1 if iter_ >= maxiter else 0,
        "time": exe_time,
        "message": msg,
        "diagnostic": log,
    }

    return res


class des_tuner_wrapper(object):
    def __init__(self, evaluation_fc, start_config: dict, search_config: dict) -> None:
        """
        config -> dict: {HP_name: (lower, upper)}
        """
        self.eval_fc = evaluation_fc
        self.search_config = search_config
        self.default_config = start_config
        self.hp_tuned = search_config.keys()

    def fit(self, kwargs: dict):
        result = des_classic(
            np.array([self.default_config[hp] for hp in self.hp_tuned]),
            self.eval_fc,
            upper=np.array([self.search_config[hp][1] for hp in self.hp_tuned]),
            lower=np.array([self.search_config[hp][0] for hp in self.hp_tuned]),
            **kwargs
        )
        result["hp_names"] = self.hp_tuned
        return result


# Example usage:
if __name__ == "__main__":
    par = [-100, -100, -100, -100]
    fn = (
        lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2
    )  # Example fitness function
    kwargs = {
        "upper": np.array([-5, -23.3, 14, 11]),
        "lower": np.array([-101, -101, -101, -150]),
        "stopfitness": 1e-10,
        "lambda": 100,
        "time": 5,
    }
    result = des_classic(par, fn, **kwargs)
    print(result)
    result = des_tuner_wrapper(
        fn,
        {"x1": -50, "x2": -20, "x3": -100, "x4": 10},
        {"x1": (-101, -5), "x2": (-101, -23.3), "x3": (-101, 14), "x4": (3, 11)},
    ).fit({"stopfitness": 1e-10,"lambda": 100,"time": 5,})
    print(result)
