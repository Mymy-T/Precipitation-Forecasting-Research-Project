import os
import sys
from performance_log_manager import PerformanceLogManager, ExperimentType
import variables

def main():
    # Ensure the use runs this file from the right directory
    terminal_dir = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if terminal_dir != script_dir:
        sys.exit(
            f"ERROR: Script executed from the wrong folder.\n"
            f"Current working directory: {terminal_dir}\n"
            f"Expected working directory: {script_dir}"
        )
    
    test_hyperparams()

def run_hyperparmeter_test(nums_locations, dists, hyperparam_test_config, performance_log_manager):
    # Hyperparam selection done on valid f1
    # draw_num and train_seed both set to 1; they don't matter here
    draw_num = 1
    train_seed = 1
    control_dist = 50 # control_dist also doesn't matter
    control_run_name_prefix = f"hparam_tune__1_sites"

    # Tune the lookback
    lookback_results = {}
    for lookback in hyperparam_test_config.LOOKBACKS:
        f1_scores = []
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=lookback, num_hiddens=hyperparam_test_config.DEFAULT_NUM_HIDDENS, weight_decay=hyperparam_test_config.DEFAULT_WEIGHT_DECAY, num_locations=1, dist=control_dist, draw_num=draw_num, train_seed=train_seed, run_name=control_run_name_prefix+f"__lb{lookback}__nh{hyperparam_test_config.DEFAULT_NUM_HIDDENS}__wd{hyperparam_test_config.DEFAULT_WEIGHT_DECAY}")
        f1_scores.append(valid_f1_score)
        for num_locations in nums_locations:
            for dist in dists:
                run_name_prefix = f"hparam_tune__{num_locations}_sites__{dist}_miles"
                _, valid_f1_score = performance_log_manager.run_and_log(lookback=lookback, num_hiddens=hyperparam_test_config.DEFAULT_NUM_HIDDENS, weight_decay=hyperparam_test_config.DEFAULT_WEIGHT_DECAY, num_locations=num_locations, dist=dist, draw_num=draw_num, train_seed=train_seed, run_name=run_name_prefix+f"__lb{lookback}__nh{hyperparam_test_config.DEFAULT_NUM_HIDDENS}__wd{hyperparam_test_config.DEFAULT_WEIGHT_DECAY}")
                f1_scores.append(valid_f1_score)
        lookback_results[lookback] = sum(f1_scores) / len(f1_scores) # Mean over all locations & dist bounds
    best_lookback = max(lookback_results, key=lookback_results.get)
    print(f"Best lookback appears to be {best_lookback}, with valid f1 score of {lookback_results[best_lookback]:.6f}")

    # Tune num_hiddens
    num_hiddens_results = {}
    for num_hiddens in hyperparam_test_config.NUM_HIDDENS_OPTIONS:
        f1_scores = []
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=num_hiddens, weight_decay=hyperparam_test_config.DEFAULT_WEIGHT_DECAY, num_locations=1, dist=control_dist, draw_num=draw_num, train_seed=train_seed, run_name=control_run_name_prefix + f"__lb{best_lookback}__nh{num_hiddens}__wd{hyperparam_test_config.DEFAULT_WEIGHT_DECAY}")
        f1_scores.append(valid_f1_score)
        for num_locations in nums_locations:
            for dist in dists:
                run_name_prefix = f"hparam_tune__{num_locations}_sites__{dist}_miles"
                _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=num_hiddens, weight_decay=hyperparam_test_config.DEFAULT_WEIGHT_DECAY, num_locations=num_locations, dist=dist, draw_num=draw_num, train_seed=train_seed, run_name=run_name_prefix + f"__lb{best_lookback}__nh{num_hiddens}__wd{hyperparam_test_config.DEFAULT_WEIGHT_DECAY}")
                f1_scores.append(valid_f1_score)
        num_hiddens_results[num_hiddens] = sum(f1_scores) / len(f1_scores) # Mean over all locations & dist bounds
    best_num_hiddens = max(num_hiddens_results, key=num_hiddens_results.get)
    print(f"Best num_hiddens appears to be {best_num_hiddens}, with valid f1 score of {num_hiddens_results[best_num_hiddens]:.6f}")

    weight_decay_results = {}
    for weight_decay in hyperparam_test_config.WEIGHT_DECAY_OPTIONS:
        f1_scores = []
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=best_num_hiddens, weight_decay=weight_decay, num_locations=1, dist=control_dist, draw_num=draw_num, train_seed=train_seed, run_name=control_run_name_prefix + f"__lb{best_lookback}__nh{best_num_hiddens}__wd{weight_decay}")
        f1_scores.append(valid_f1_score)
        for num_locations in nums_locations:
            for dist in dists:
                run_name_prefix = f"hparam_tune__{num_locations}_sites__{dist}_miles"
                _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=best_num_hiddens, weight_decay=weight_decay, num_locations=num_locations, dist=dist, draw_num=draw_num, train_seed=train_seed, run_name=run_name_prefix + f"__lb{best_lookback}__nh{best_num_hiddens}__wd{weight_decay}")
                f1_scores.append(valid_f1_score)
        weight_decay_results[weight_decay] = sum(f1_scores) / len(f1_scores) # Mean over all locations & dist bounds
    best_weight_decay = max(weight_decay_results, key=weight_decay_results.get)
    print(f"Best weight_decay appears to be {best_weight_decay}, with valid f1 score of {weight_decay_results[best_weight_decay]:.6f}")

    return best_lookback, best_num_hiddens, best_weight_decay


def test_hyperparams():
    hyperparam_test_config = variables.HyperparameterTestingConfig()
    experiment_config = variables.ExperimentConfig()
    performance_log_manager = PerformanceLogManager(ExperimentType.hyperparameter_testing)
    nums_locations = experiment_config.NUM_LOCATIONS
    dists = experiment_config.DISTS
    lb, nh, wd = run_hyperparmeter_test(nums_locations=nums_locations, dists=dists, hyperparam_test_config=hyperparam_test_config, performance_log_manager=performance_log_manager)
    
    print("Brief hyperparameter tuning complete!")
    print(f"Best lookback: {lb} \t Best num hiddens: {nh} \t Best weight decay: {wd:.6f}")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.MODEL_SAVE_FOLDER)} for all evaluated models.")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.RESULTS_PATH)} for all performance data.")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.RUNS_FOLDER)} for training graphs to check for underfitting/overfitting.")


if __name__ == "__main__":
    main()