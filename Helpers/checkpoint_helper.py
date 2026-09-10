import os
import glob
import torch as th


class CheckpointHelper:
    @staticmethod
    def init(params, algo_name: str, run_ts: str, keep_last: int = 5):
        from Environment.environment_utility import build_csv_name

        capacity_weight = getattr(params, "capacity_weight", None)
        miss_weight = getattr(params, "miss_weight", None)
        features = (
            f"cap{capacity_weight:g}_miss{miss_weight:g}"
            if capacity_weight is not None and miss_weight is not None
            else None
        )

        prefix = build_csv_name(
            algo_name=algo_name,
            task_type=params.task_type,
            n_agent=int(params.n_agent),
            n_sc=int(params.n_sc),
            ff_tag=getattr(params, "fast_fading_tag", "FF"),
            trial_run=params.trial_run,
            ts=run_ts,
            loc=params.loc,
            features=features,
        )[:-4]  # strip ".csv", reuse the same identifying prefix for checkpoints

        out_dir = os.path.join("Results", algo_name, "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        return {"out_dir": out_dir, "prefix": prefix, "keep_last": keep_last}

    @staticmethod
    def init_from_prefix(prefix: str, algo_name: str, keep_last: int = 5):
        """Resume variant of init(): reuses an existing run's exact prefix
        (recovered via prefix_from_checkpoint_path) instead of building a new
        one, so continued checkpoints land in the same lineage."""
        out_dir = os.path.join("Results", algo_name, "checkpoints")
        os.makedirs(out_dir, exist_ok=True)
        return {"out_dir": out_dir, "prefix": prefix, "keep_last": keep_last}

    @staticmethod
    def save(ckpt_ctx: dict, models: dict, episode: int, extra_state: dict = None):
        out_dir = ckpt_ctx["out_dir"]
        prefix = ckpt_ctx["prefix"]
        keep_last = ckpt_ctx["keep_last"]

        path = os.path.join(out_dir, f"{prefix}_ep{episode}.pt")
        state = {"episode": episode}
        for name, obj in models.items():
            state[name] = obj.state_dict()
        if extra_state:
            state.update(extra_state)
        th.save(state, path)

        existing = sorted(
            glob.glob(os.path.join(out_dir, f"{prefix}_ep*.pt")),
            key=os.path.getmtime,
        )
        for stale_path in existing[:-keep_last]:
            os.remove(stale_path)

    @staticmethod
    def load(path: str, map_location=None):
        return th.load(path, map_location=map_location)

    @staticmethod
    def prefix_from_checkpoint_path(path: str) -> str:
        """Recover the run's {prefix} (matching its CSV filename minus '.csv')
        from one of its '{prefix}_ep{episode}.pt' checkpoint files."""
        name = os.path.splitext(os.path.basename(path))[0]
        idx = name.rfind("_ep")
        if idx == -1:
            raise ValueError(f"Not a checkpoint filename (no '_ep<N>' suffix): {path}")
        return name[:idx]
