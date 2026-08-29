import csv
import os
import sys

import torch
from torch.utils.data import DataLoader

from dataset import DatasetType, PrecipitationClassificationDataset
from trainer import compute_confusion_matrix, compute_f1, compute_precision_recall
import variables


def build_test_dataloader():
    data_config = variables.DataConfig()
    main_config = variables.MainConfig()
    fixed_hyperparam_config = variables.FixedHyperparameterConfig()
    file_label = variables.generate_file_label(num_locations=1)
    test_ds = PrecipitationClassificationDataset(
        folder=data_config.PROCESSED_CSV_FOLDER_NAME,
        file_label=file_label,
        lookback=main_config.LOOKBACK,
        dataset_type=DatasetType.test,
    )
    running_on_cpu = variables.DEVICE.type == "cpu"
    return DataLoader(test_ds, batch_size=fixed_hyperparam_config.BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=(not running_on_cpu))


def run_random_guessing(test_dataloader, seed):
    """One replicate of the random-guessing baseline; seeded for reproducibility."""
    torch.manual_seed(seed)
    total_examples = 0
    fn_tot = fp_tot = tn_tot = tp_tot = 0
    for _, y in test_dataloader:
        y = y.to(variables.DEVICE)
        prob_pred = torch.rand_like(y)
        total_examples += prob_pred.shape[0]
        fn, fp, tn, tp = compute_confusion_matrix(prob_pred, y)
        fn_tot += fn
        fp_tot += fp
        tn_tot += tn
        tp_tot += tp
    f1 = compute_f1(tp_tot, fn_tot, fp_tot)
    precision, recall = compute_precision_recall(tp_tot, fn_tot, fp_tot)
    return f1, tp_tot, fp_tot, tn_tot, fn_tot, precision, recall


def main():
    terminal_dir = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if terminal_dir != script_dir:
        sys.exit(
            f"ERROR: Script executed from the wrong folder.\n"
            f"Current working directory: {terminal_dir}\n"
            f"Expected working directory: {script_dir}"
        )

    random_guessing_config = variables.RandomGuessingConfig()

    test_dataloader = build_test_dataloader()
    rows = []
    for run_num, seed in enumerate(random_guessing_config.SEEDS[: random_guessing_config.NUM_RUNS], start=1):
        f1, tp, fp, tn, fn, precision, recall = run_random_guessing(test_dataloader, seed)
        print(f"Run {run_num} (seed={seed}): F1={f1:.4f}, precision={precision:.4f}, recall={recall:.4f}, TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        rows.append(
            {
                "run": run_num,
                "seed": seed,
                "F1": f1,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "precision": precision,
                "recall": recall,
            }
        )

    with open(os.path.join(os.getcwd(), random_guessing_config.RESULTS_PATH), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=random_guessing_config.RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {random_guessing_config.RESULTS_PATH}")


if __name__ == "__main__":
    main()
