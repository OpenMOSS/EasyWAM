from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from experiments.libero_pro.libero_pro_utils import (
    OFFICIAL_MAX_STEPS,
    PROTOCOL_ID,
    TaskSpec,
    is_valid_result,
    prepare_runtime_config,
    registered_suite_name,
    result_path,
    select_tasks,
    write_jsonl,
)
from experiments.libero_pro.summarize_libero_pro import summarize_results


def _task(
    *,
    base_suite: str = "libero_goal",
    perturbation: str = "object",
    task_id: int = 0,
) -> TaskSpec:
    return TaskSpec(
        base_suite=base_suite,
        suite=registered_suite_name(base_suite, perturbation),
        perturbation=perturbation,
        task_id=task_id,
        task_name=f"task_{task_id}",
        num_trials=50,
        max_steps=OFFICIAL_MAX_STEPS[base_suite],
        protocol=PROTOCOL_ID,
        source_commit="test-commit",
    )


class LiberoProUtilsTest(unittest.TestCase):
    def test_official_suite_mapping(self) -> None:
        self.assertEqual(
            registered_suite_name("libero_spatial", "position"),
            "libero_spatial_swap",
        )
        self.assertEqual(
            registered_suite_name("libero_10", "semantic"),
            "libero_10_lan",
        )

    def test_task_filter_applies_to_every_suite(self) -> None:
        first = [_task(task_id=index) for index in range(10)]
        second = [
            _task(base_suite="libero_object", perturbation="task", task_id=index)
            for index in range(10)
        ]
        selected = select_tasks(
            {first[0].suite: first, second[0].suite: second}, task_ids=[1, 3]
        )
        self.assertEqual([task.task_id for task in selected], [1, 3, 1, 3])

    def test_runtime_config_keeps_source_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            data = root / "data"
            config_dir = root / "runtime"
            (source / "libero/libero/benchmark").mkdir(parents=True)
            (source / "libero/libero/assets").mkdir(parents=True)
            (source / "libero/libero/__init__.py").write_text("", encoding="utf-8")
            (source / "libero/libero/benchmark/__init__.py").write_text(
                "", encoding="utf-8"
            )
            (data / "bddl_files").mkdir(parents=True)
            (data / "init_files").mkdir(parents=True)
            before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))
            config_path = prepare_runtime_config(source, data, config_dir)
            after = sorted(str(path.relative_to(source)) for path in source.rglob("*"))
            self.assertEqual(before, after)
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(config["bddl_files"]), data / "bddl_files")
            self.assertEqual(Path(config["assets"]), source / "libero/libero/assets")

    def test_result_validation_checks_protocol_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            task = _task()
            path = result_path(output, task)
            path.parent.mkdir(parents=True)
            payload = {
                "task_suite": task.suite,
                "base_suite": task.base_suite,
                "perturbation": task.perturbation,
                "task_id": task.task_id,
                "task_name": task.task_name,
                "protocol": task.protocol,
                "source_commit": task.source_commit,
                "total_episodes": 50,
                "successes": 25,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(is_valid_result(path, task))
            payload["total_episodes"] = 10
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(is_valid_result(path, task))

    def test_summary_writes_normalized_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tasks = [
                _task(base_suite="libero_goal", perturbation="object"),
                _task(base_suite="libero_goal", perturbation="semantic"),
            ]
            write_jsonl(output / "tasks.jsonl", (task.to_dict() for task in tasks))
            for task, successes in zip(tasks, (25, 50)):
                path = result_path(output, task)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "task_suite": task.suite,
                            "base_suite": task.base_suite,
                            "perturbation": task.perturbation,
                            "task_id": task.task_id,
                            "task_name": task.task_name,
                            "protocol": task.protocol,
                            "source_commit": task.source_commit,
                            "task_description": task.task_name,
                            "total_episodes": 50,
                            "successes": successes,
                            "duration": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
            summary = summarize_results(output)
            self.assertTrue(summary["leaderboard_complete"])
            self.assertAlmostEqual(summary["leaderboard_average"], 0.75)
            leaderboard = (output / "leaderboard.csv").read_text(encoding="utf-8")
            self.assertIn("0.5", leaderboard)
            self.assertIn("1.0", leaderboard)

    def test_environment_language_is_used_as_prompt(self) -> None:
        from experiments.libero import libero_utils

        fake_env = SimpleNamespace(
            language_instruction="grasp the crimson container",
            close=mock.Mock(),
        )
        task = SimpleNamespace(
            language="pick up the red bowl",
            problem_folder="libero_goal_lan",
            bddl_file="task.bddl",
        )
        with mock.patch.object(
            libero_utils, "get_libero_path", return_value="/data"
        ), mock.patch.object(libero_utils, "LiberoEnvProcess", return_value=fake_env):
            env, description = libero_utils.get_libero_env(
                task, 256, 7, prompt_source="environment_language"
            )
        self.assertIs(env, fake_env)
        self.assertEqual(description, "grasp the crimson container")


if __name__ == "__main__":
    unittest.main()
