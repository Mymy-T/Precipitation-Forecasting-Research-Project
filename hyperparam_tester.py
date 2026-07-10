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

def run_hyperparmeter_test(num_locations, hyperparam_test_config, performance_log_manager):
    
    # For 1 location, latter params ignored; for 10 locations, any dist, draw_num, and train_seed will do
    # Selection is done on valid_f1 to avoid data leakage
    lookback_results = {}
    for lookback in hyperparam_test_config.LOOKBACKS:
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=lookback, num_hiddens=64, weight_decay=0, num_locations=num_locations, dist=50, draw_num=1, train_seed=1, run_name=f"{num_locations}_sites__{lookback}__64__0")
        lookback_results[lookback] = valid_f1_score
    best_lookback = max(lookback_results, key=lookback_results.get)
    print(f"For {num_locations} location(s), best lookback appears to be {best_lookback}, with valid f1 score of {lookback_results[best_lookback]:.4f}")

    num_hiddens_results = {}
    for num_hiddens in hyperparam_test_config.NUM_HIDDENS_OPTIONS:
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=num_hiddens, weight_decay=0, num_locations=num_locations, dist=50, draw_num=1, train_seed=1, run_name=f"{num_locations}_sites__{best_lookback}__{num_hiddens}__0")
        num_hiddens_results[num_hiddens] = valid_f1_score
    best_num_hiddens = max(num_hiddens_results, key=num_hiddens_results.get)
    print(f"For {num_locations} location(s), best num_hiddens appears to be {best_num_hiddens}, with valid f1 score of {num_hiddens_results[best_num_hiddens]:.4f}")

    weight_decay_results = {}
    for weight_decay in hyperparam_test_config.WEIGHT_DECAY_OPTIONS:
        _, valid_f1_score = performance_log_manager.run_and_log(lookback=best_lookback, num_hiddens=best_num_hiddens, weight_decay=weight_decay, num_locations=num_locations, dist=50, draw_num=1, train_seed=1, run_name=f"{num_locations}_sites__{best_lookback}__{best_num_hiddens}__{weight_decay}")
        weight_decay_results[weight_decay] = valid_f1_score
    best_weight_decay = max(weight_decay_results, key=weight_decay_results.get)
    print(f"For {num_locations} location(s), best weight_decay appears to be {best_weight_decay}, with valid f1 score of {weight_decay_results[best_weight_decay]:.4f}")


def test_hyperparams():
    hyperparam_test_config = variables.HyperparameterTestingConfig()
    performance_log_manager = PerformanceLogManager(ExperimentType.hyperparameter_testing)

    run_hyperparmeter_test(num_locations=1, hyperparam_test_config=hyperparam_test_config, performance_log_manager=performance_log_manager)
    run_hyperparmeter_test(num_locations=10, hyperparam_test_config=hyperparam_test_config, performance_log_manager=performance_log_manager)

    print("Brief hyperparameter tuning complete!")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.MODEL_SAVE_FOLDER)} for all evaluated models.")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.RESULTS_PATH)} for all performance data.")
    print(f"See {os.path.join(os.getcwd(), hyperparam_test_config.RUNS_FOLDER)} for training graphs to check for underfitting/overfitting.")


if __name__ == "__main__":
    main()