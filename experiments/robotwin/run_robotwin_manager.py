"""Parallel task-level RoboTwin evaluator for the WING policy adapter."""

from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvalJob:
    task_name: str
    task_config: str
    seed: int
    output_dir: str

    @property
    def key(self) -> str:
        return f"{self.task_name}/{self.task_config}/seed_{self.seed}"


def discover_robotwin_tasks(robotwin_root: Path) -> list[str]:
    instruction_dir = robotwin_root / "description" / "task_instruction"
    env_dir = robotwin_root / "envs"
    tasks = [
        path.stem
        for path in instruction_dir.glob("*.json")
        if (env_dir / f"{path.stem}.py").is_file()
    ]
    if not tasks:
        raise FileNotFoundError(
            f"No RoboTwin tasks found under {instruction_dir} and {env_dir}."
        )
    return sorted(tasks)


def build_jobs(
    *,
    task_names: list[str],
    task_configs: list[str],
    seeds: list[int],
    output_root: Path,
) -> list[EvalJob]:
    jobs = []
    for task_name in task_names:
        for task_config in task_configs:
            for seed in seeds:
                jobs.append(
                    EvalJob(
                        task_name=str(task_name),
                        task_config=str(task_config),
                        seed=int(seed),
                        output_dir=str(
                            output_root / str(task_name) / str(task_config) / f"seed_{int(seed)}"
                        ),
                    )
                )
    return jobs


def _worker_payload(cfg: DictConfig, job: EvalJob, task_choice: str) -> dict[str, Any]:
    evaluation = cfg.EVALUATION
    return {
        "task_name": job.task_name,
        "task_config": job.task_config,
        "ckpt_setting": Path(str(cfg.ckpt)).stem,
        "policy_name": "wing_policy",
        "instruction_type": str(evaluation.instruction_type),
        "seed": int(job.seed),
        "eval_num_episodes": int(evaluation.eval_num_episodes),
        "eval_output_dir": job.output_dir,
        "skip_get_obs_within_replan": bool(evaluation.skip_get_obs_within_replan),
        "wing_task": task_choice,
        "wing_overrides": [str(x) for x in evaluation.get("wing_overrides", [])],
        "ckpt": str(cfg.ckpt),
        "dataset_stats_path": str(evaluation.dataset_stats_path),
        "device": "cuda:0",
        "mixed_precision": str(cfg.get("mixed_precision", "bf16")),
        "action_horizon": evaluation.get("action_horizon"),
        "replan_steps": int(evaluation.replan_steps),
        "num_inference_steps": int(evaluation.num_inference_steps),
        "sigma_shift": evaluation.get("sigma_shift"),
        "text_cfg_scale": float(evaluation.text_cfg_scale),
        "negative_prompt": str(evaluation.negative_prompt),
        "rand_device": str(evaluation.rand_device),
        "tiled": bool(evaluation.tiled),
    }


def _run_job(
    *,
    cfg: DictConfig,
    job: EvalJob,
    gpu_id: int,
    task_choice: str,
    robotwin_root: Path,
) -> dict[str, Any]:
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_config = output_dir / "worker_config.yaml"
    worker_config.write_text(
        yaml.safe_dump(_worker_payload(cfg, job, task_choice), sort_keys=False),
        encoding="utf-8",
    )
    log_path = output_dir / "worker.log"
    result_path = output_dir / "result.json"
    # A deterministic job directory may be reused for a retry. Never accept a
    # result left by an earlier invocation as the result of this worker.
    result_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    python_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT / "experiments" / "robotwin")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    command = [
        str(cfg.MULTIRUN.get("python_bin") or sys.executable),
        "script/eval_policy.py",
        "--config",
        str(worker_config),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=robotwin_root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    result = {
        **asdict(job),
        "key": job.key,
        "gpu_id": int(gpu_id),
        "returncode": int(completed.returncode),
        "duration": time.time() - started,
        "log": str(log_path),
    }
    if completed.returncode == 0 and result_path.is_file():
        result.update(json.loads(result_path.read_text(encoding="utf-8")))
    else:
        result["error"] = "worker failed" if completed.returncode else "missing result.json"
    return result


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: str(item["key"]))
    failures = [item for item in ordered if int(item.get("returncode", 1)) != 0 or "error" in item]
    successes = sum(int(item.get("successes", 0)) for item in ordered if item not in failures)
    episodes = sum(int(item.get("total_episodes", 0)) for item in ordered if item not in failures)
    summary = {
        "jobs": len(ordered),
        "completed_jobs": len(ordered) - len(failures),
        "failed_jobs": len(failures),
        "successes": successes,
        "total_episodes": episodes,
        "success_rate": None if episodes == 0 else successes / episodes,
        "results": ordered,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_name", "task_config", "seed", "gpu_id", "successes",
                "total_episodes", "success_rate", "duration", "returncode", "log",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(ordered)
    return summary


def run_parallel(cfg: DictConfig, jobs: list[EvalJob], task_choice: str) -> dict[str, Any]:
    robotwin_root = Path(str(cfg.EVALUATION.robotwin_root)).expanduser().resolve()
    eval_script = robotwin_root / "script" / "eval_policy.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"RoboTwin evaluator not found: {eval_script}")
    for task_config in sorted({job.task_config for job in jobs}):
        path = robotwin_root / "task_config" / f"{task_config}.yml"
        if not path.is_file():
            raise FileNotFoundError(
                f"RoboTwin task config not found: {path}. Complete the official asset/config install."
            )

    gpu_ids = [int(x) for x in cfg.MULTIRUN.gpu_ids]
    max_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if not gpu_ids or max_per_gpu <= 0:
        raise ValueError("MULTIRUN.gpu_ids and max_tasks_per_gpu must provide at least one slot.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"MULTIRUN.gpu_ids must not contain duplicates: {gpu_ids}.")
    slots: queue.Queue[int] = queue.Queue()
    for gpu_id in gpu_ids:
        for _ in range(max_per_gpu):
            slots.put(gpu_id)

    def execute(job: EvalJob) -> dict[str, Any]:
        gpu_id = slots.get()
        try:
            return _run_job(
                cfg=cfg,
                job=job,
                gpu_id=gpu_id,
                task_choice=task_choice,
                robotwin_root=robotwin_root,
            )
        finally:
            slots.put(gpu_id)

    results = []
    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{len(results)}/{len(jobs)}] {result['key']} gpu={result['gpu_id']} "
                f"returncode={result['returncode']}"
            )
    return write_summary(Path(str(cfg.EVALUATION.output_dir)), results)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None or cfg.EVALUATION.dataset_stats_path is None:
        raise ValueError("ckpt and EVALUATION.dataset_stats_path are required.")
    task_choice = str(HydraConfig.get().runtime.choices["task"])
    output_root = Path(str(cfg.EVALUATION.output_dir)).expanduser().resolve()
    robotwin_root = Path(str(cfg.EVALUATION.robotwin_root)).expanduser().resolve()
    configured_tasks = [str(x) for x in cfg.MULTIRUN.task_names]
    task_names = configured_tasks or discover_robotwin_tasks(robotwin_root)
    jobs = build_jobs(
        task_names=task_names,
        task_configs=[str(x) for x in cfg.MULTIRUN.task_configs],
        seeds=[int(x) for x in cfg.MULTIRUN.seeds],
        output_root=output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "jobs.json").write_text(
        json.dumps([asdict(job) for job in jobs], indent=2), encoding="utf-8"
    )
    OmegaConf.save(cfg, output_root / "manager_config.yaml")
    if bool(cfg.MULTIRUN.create_only):
        print(f"Created {len(jobs)} RoboTwin jobs at {output_root / 'jobs.json'}")
        return
    summary = run_parallel(cfg, jobs, task_choice)
    if summary["failed_jobs"]:
        raise RuntimeError(
            f"{summary['failed_jobs']} RoboTwin jobs failed; see {output_root / 'summary.json'}."
        )


if __name__ == "__main__":
    main()
