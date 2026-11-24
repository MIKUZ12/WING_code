from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalJob:
    suite: str
    task_id: int


def _worker_overrides() -> list[str]:
    blocked = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
    }
    result = []
    for override in HydraConfig.get().overrides.task:
        key = override.split("=", 1)[0].lstrip("+~")
        if key not in blocked and not key.startswith(("MULTIRUN.", "hydra.")):
            result.append(override)
    return result


def _jobs(suites: list[str]) -> list[EvalJob]:
    benchmark_types = benchmark.get_benchmark_dict()
    jobs = []
    for suite_name in suites:
        suite = benchmark_types[suite_name]()
        jobs.extend(EvalJob(suite_name, task_id) for task_id in range(int(suite.n_tasks)))
    return jobs


def _run(
    job: EvalJob,
    *,
    gpu_id: int,
    cfg: DictConfig,
    task_choice: str,
    overrides: list[str],
) -> dict:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    python_path = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments/libero/eval_libero_single.py"),
        f"task={task_choice}",
        f"ckpt={cfg.ckpt}",
        "gpu_id=0",
        f"EVALUATION.task_suite_name={job.suite}",
        f"EVALUATION.task_id={job.task_id}",
        f"EVALUATION.num_trials={int(cfg.EVALUATION.num_trials)}",
        f"EVALUATION.output_dir={cfg.EVALUATION.output_dir}",
        "EVALUATION.device=cuda:0",
        *overrides,
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    result_path = (
        Path(str(cfg.EVALUATION.output_dir))
        / job.suite
        / f"gpu0_task{job.task_id}_results.json"
    )
    result = asdict(job) | {"gpu_id": gpu_id, "returncode": completed.returncode}
    if completed.returncode == 0 and result_path.is_file():
        result.update(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        result["error"] = "worker failed" if completed.returncode else "missing result file"
    return result


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt is required.")
    task_choice = str(HydraConfig.get().runtime.choices["task"])
    jobs = _jobs([str(value) for value in cfg.MULTIRUN.task_suite_names])
    output_root = Path(str(cfg.EVALUATION.output_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_root / "manager_config.yaml")
    (output_root / "jobs.json").write_text(
        json.dumps([asdict(job) for job in jobs], indent=2), encoding="utf-8"
    )
    if bool(cfg.MULTIRUN.create_only):
        return

    gpu_ids = list(range(int(cfg.MULTIRUN.num_gpus)))
    slots: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        for _ in range(int(cfg.MULTIRUN.max_tasks_per_gpu)):
            slots.put(gpu_id)

    def execute(job: EvalJob) -> dict:
        gpu_id = slots.get()
        try:
            return _run(
                job,
                gpu_id=gpu_id,
                cfg=cfg,
                task_choice=task_choice,
                overrides=_worker_overrides(),
            )
        finally:
            slots.put(gpu_id)

    results = []
    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())
    failures = [result for result in results if result.get("returncode") != 0]
    summary = {
        "jobs": len(results),
        "failed_jobs": len(failures),
        "successes": sum(int(result.get("successes", 0)) for result in results),
        "episodes": sum(int(result.get("total_episodes", 0)) for result in results),
        "results": sorted(results, key=lambda item: (item["suite"], item["task_id"])),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError(f"{len(failures)} LIBERO evaluation workers failed.")


if __name__ == "__main__":
    main()
