#!/usr/bin/env python3
"""
Ray Tune script for hyperparameter sweeping YOLO models (11n & 11M)
on single-class plankton detection...
After the run it also saves:
 - fitness_history.csv,
 - trial_history.csv,
 - per-hyperparam summary CSVs in summaries/,
 - fitness_over_time.png,
 - heatmap_batch_imgsz.png,
 - heatmap_model_optimizer.png,
 - environment.json, failures.json, best_config.json.
"""
import os, sys, json, time
import ray, torch, pandas as pd, numpy as np, matplotlib.pyplot as plt
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ultralytics import YOLO

# === Configuration ===
TOTAL_SAMPLES   = 150    # total hyperparameter trials desired
BATCH_SAMPLES   = 10     # max concurrent trials
EPOCHS          = 20
SEED            = 0

# Paths
ROOT            = os.getcwd()
STORAGE_PATH    = os.path.join(ROOT, "ray_results", "plankton_hpo_sweep")
ENV_JSON        = os.path.join(STORAGE_PATH, "environment.json")
FAILURES_JSON   = os.path.join(STORAGE_PATH, "failures.json")
FIT_CSV         = os.path.join(STORAGE_PATH, "fitness_history.csv")
HIST_CSV        = os.path.join(STORAGE_PATH, "trial_history.csv")
BEST_JSON       = os.path.join(STORAGE_PATH, "best_config.json")
SUMMARY_DIR     = os.path.join(STORAGE_PATH, "summaries")
PLOT_FIT        = os.path.join(STORAGE_PATH, "fitness_over_time.png")
HEAT1           = os.path.join(STORAGE_PATH, "heatmap_batch_imgsz.png")
HEAT2           = os.path.join(STORAGE_PATH, "heatmap_model_optimizer.png")

os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.makedirs(SUMMARY_DIR, exist_ok=True)

DATA_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml"

# Save environment metadata
env_info = {
    "python_version": sys.version,
    "torch_version":   torch.__version__,
    "ultralytics":     getattr(YOLO, "__version__", "unknown"),
    "ray_version":     ray.__version__,
    "cuda_available":  torch.cuda.is_available(),
    "cuda_devices":    torch.cuda.device_count(),
    "seed":            SEED
}
with open(ENV_JSON, "w") as f:
    json.dump(env_info, f, indent=2)
print(f"Wrote env metadata → {ENV_JSON}")

FIT_WEIGHTS = np.array([0.15, 0.25, 0.3, 0.3])

def train_plankton(config):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = YOLO(config["model_path"], task="detect")
    start = time.time()
    result = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        batch=config["batch"],
        imgsz=config["imgsz"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        optimizer=config["optimizer"],
        project="Training_runs_plankto_within_ray_tuner",
        exist_ok=True,
        save=True
    )
    elapsed = time.time() - start
    mem     = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)

    P, R, m50, m5095 = result.box.mean_results()
    fitness         = float(np.dot(FIT_WEIGHTS, [P, R, m50, m5095]))

    tune.report({
        "fitness":      fitness,
        "precision":    P,
        "recall":       R,
        "mAP50":        m50,
        "mAP50-95":     m5095,
        "elapsed_time": elapsed,
        "memory_mb":    mem,
    })

if __name__ == "__main__":
    ray.init()

    search_space = {
        "model_path": tune.choice(["yolo11n.pt","yolo11m.pt"]),
        "optimizer":  tune.choice(["Adam","SGD","AdamW"]),
        "batch":      tune.choice([4,8,16,32]),
        "imgsz":      tune.choice([640,960,1024,1280]),
        "lr0":        tune.loguniform(1e-4,1e-2),
        "lrf":        tune.loguniform(1e-4,1e-2)
    }

    scheduler = ASHAScheduler(
        #metric="fitness", mode="max",
        max_t=EPOCHS, grace_period=4, reduction_factor=4
    )

    analysis = tune.run(
        train_plankton,
        config=search_space,
        num_samples=TOTAL_SAMPLES,
        max_concurrent_trials=BATCH_SAMPLES,
        scheduler=scheduler,
        resources_per_trial={"cpu":4,"gpu":1},
        metric="fitness",
        mode="max",
        storage_path=STORAGE_PATH,
        name="plankton_hpo",
        raise_on_failed_trial=False
    )

    # === 1) Raw fitness per trial ===
    results_df = analysis.results_df
    results_df.to_csv(FIT_CSV, index=False)
    print(f"Saved trial‐level fitness → {FIT_CSV}")

    # === 2) Per‐epoch histories ===
    # `analysis.trial_dataframes` is a dict {trial_id: df}
    hist_dfs = []
    for tid, df in analysis.trial_dataframes.items():
        df2 = df.copy()
        df2["trial_id"] = tid
        hist_dfs.append(df2)
    history_df = pd.concat(hist_dfs, ignore_index=True)
    history_df.to_csv(HIST_CSV, index=False)
    print(f"Saved epoch‐level history → {HIST_CSV}")

    # === 3) Failures & best config ===
    failures = []
    for t in analysis.trials:
        if t.status == t.ERROR:
            err = t.get_error() or repr(t.get_pickled_error())
            failures.append({
                "trial_id":t.trial_id,
                "config":  t.config,
                "error":   err
            })
    with open(FAILURES_JSON, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"Saved failures → {FAILURES_JSON}")

    best_trial  = analysis.get_best_trial(metric="fitness", mode="max", scope="all")
    if best_trial:
        # get the hyperparameter dict
        best_cfg = best_trial.config.copy()
        # pull out its final fitness
        best_cfg["fitness"] = best_trial.last_result["fitness"]
        # save it
        with open(BEST_JSON, "w") as f:
            json.dump(best_cfg, f, indent=2)
        print(f"Saved best config → {BEST_JSON}")
    else:
        print("No successful trials—cannot pick a best config.")

    # Only do per‐hyperparam summaries if there is at least one successful trial
    if not results_df.empty:
        def save_group(col, fname):
            g = results_df.groupby(col)["fitness"].agg(["mean", "count"])
            out = os.path.join(SUMMARY_DIR, fname)
            g.to_csv(out)
            print(f"Summary by {col} → {out}")
            return g

        save_group("config/model_path", "by_model.csv")
        save_group("config/optimizer",  "by_optimizer.csv")
        save_group("config/batch",      "by_batch.csv")
        save_group("config/imgsz",      "by_imgsz.csv")
    else:
        print("No successful trials → skipping per‐hyperparam summaries.")

    # Only plot fitness‐over‐time if we actually collected any history
    if not history_df.empty and "fitness" in history_df and "training_iteration" in history_df:
        plt.figure()
        for _, grp in history_df.groupby("trial_id"):
            plt.plot(grp["training_iteration"], grp["fitness"], alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Fitness")
        plt.title("Fitness over time")
        plt.savefig(PLOT_FIT)
        plt.close()
        print(f"Saved plot → {PLOT_FIT}")
    else:
        print("No per‐epoch history → skipping fitness‐over‐time plot.")

    # Only make each heatmap if there’s data to pivot
    if not results_df.empty:
        pivot = results_df.pivot_table(
            index="config/batch",
            columns="config/imgsz",
            values="fitness",
            aggfunc="mean"
        )
        if not pivot.empty:
            plt.figure()
            plt.imshow(pivot, aspect="auto")
            plt.colorbar(label="Mean fitness")
            plt.xlabel("imgsz"); plt.ylabel("batch")
            plt.title("Batch vs imgsz")
            plt.savefig(HEAT1)
            plt.close()
            print(f"Saved heatmap → {HEAT1}")
        else:
            print("Pivot (batch vs imgsz) is empty → skipping that heatmap.")

        pivot2 = results_df.pivot_table(
            index="config/model_path",
            columns="config/optimizer",
            values="fitness",
            aggfunc="mean"
        )
        if not pivot2.empty:
            plt.figure()
            plt.imshow(pivot2, aspect="auto")
            plt.colorbar(label="Mean fitness")
            plt.xlabel("optimizer"); plt.ylabel("model")
            plt.title("Model vs Optimizer")
            plt.savefig(HEAT2)
            plt.close()
            print(f"Saved heatmap → {HEAT2}")
        else:
            print("Pivot (model vs optimizer) is empty → skipping that heatmap.")
    else:
        print("No successful trials → skipping heatmaps.")