import os
from performance_log_manager import PerformanceLogManager, ExperimentType
import sys
import variables

def main():
    # Ensure the user runs this file from the right directory
    terminal_dir = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if terminal_dir != script_dir:
        sys.exit(
            f"ERROR: Script executed from the wrong folder.\n"
            f"Current working directory: {terminal_dir}\n"
            f"Expected working directory: {script_dir}"
        )
    
    run_all_combinations()


def run_all_combinations():
    experiment_config = variables.ExperimentConfig()
    main_config = variables.MainConfig()
    performance_log_manager = PerformanceLogManager(experiment_type=ExperimentType.main)

    lookback = main_config.LOOKBACK
    num_hiddens = main_config.NUM_HIDDENS
    weight_decay = main_config.WEIGHT_DECAY

    control_file_label = variables.generate_file_label(num_locations=1)
    for train_seed in experiment_config.TRAINING_SEEDS: # Run for control (n_locations = 1)
        performance_log_manager.run_and_log(lookback=lookback, num_hiddens=num_hiddens, weight_decay=weight_decay, num_locations=1, dist=0, draw_num=1, train_seed=train_seed, run_name=f"{control_file_label}__seed{train_seed}")

    for dist in experiment_config.DISTS: # Run for all other combinations
        for num_locations in experiment_config.NUM_LOCATIONS:
            for draw_num in range(1, experiment_config.NUM_DRAWS + 1):
                file_label = variables.generate_file_label(num_locations=num_locations, dist=dist, draw_num=draw_num)
                for train_seed in experiment_config.TRAINING_SEEDS:
                    run_name = f"{file_label}__seed{train_seed}"
                    performance_log_manager.run_and_log(lookback, num_hiddens, weight_decay, num_locations, dist, draw_num, train_seed, run_name)

if __name__ == "__main__":
    main()