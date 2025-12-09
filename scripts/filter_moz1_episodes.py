#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# 根据 moz1 数据集的 event_log.jsonl 中的 payload.is_mistake 字段，
# 过滤出成功（is_mistake == False）的 episode，并生成新的 episodes.jsonl。
#
# 使用方式（示例）：
#   python scripts/filter_moz1_episodes.py \
#       --dataset-root /path/to/moz1_folding_1130/snapshots/<snapshot_id> \
#       --overwrite True
#
# 默认会在 meta 目录下生成一个 episodes_success.jsonl，
# 如果传入 --overwrite True，会先备份原 episodes.jsonl 为 episodes_all_backup.jsonl，
# 然后用过滤后的 episodes_success.jsonl 覆盖 episodes.jsonl。

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import tyro


@dataclass
class FilterConfig:
    # snapshot 根目录（包含 meta/, data/, videos/, event_log.jsonl）
    dataset_root: str

    # event_log 路径，默认使用 dataset_root 下面的 event_log.jsonl
    event_log_path: str | None = None

    # 原始 episodes.jsonl 路径
    episodes_path: str | None = None

    # 过滤结果输出路径
    output_episodes_path: str | None = None

    # 是否覆盖原始 episodes.jsonl
    overwrite: bool = False


def collect_success_episodes(event_log_path: Path) -> set[int]:
    """
    从 event_log.jsonl 中提取成功 episode 的 episode_idx。

    逻辑：
    - 只看 endpoint == "record" 的行
    - 每行 payload 里有:
        - episode_idx: int
        - is_mistake: bool
    - 如果同一个 episode 多次被记录，最后一次记录为准。
      最终 is_mistake == False 视为成功。
    """
    print(f"[filter_moz1_episodes] 读取 event_log: {event_log_path}")

    episode_to_mistake: Dict[int, bool] = {}
    num_records = 0

    with event_log_path.open("r") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("endpoint") != "record":
                continue

            payload = obj.get("payload", {})
            if not isinstance(payload, dict):
                continue

            if "episode_idx" not in payload or "is_mistake" not in payload:
                continue

            ep_idx = payload["episode_idx"]
            is_mistake = bool(payload["is_mistake"])
            if not isinstance(ep_idx, int):
                # 安全起见，尝试转成 int
                try:
                    ep_idx = int(ep_idx)
                except Exception:
                    continue

            episode_to_mistake[ep_idx] = is_mistake
            num_records += 1

    print(
        f"[filter_moz1_episodes] 共解析到 {len(episode_to_mistake)} 个 episode 的记录，"
        f"record 行数={num_records}"
    )

    success_eps = {ep for ep, is_mistake in episode_to_mistake.items() if not is_mistake}
    failed_eps = {ep for ep, is_mistake in episode_to_mistake.items() if is_mistake}
    print(
        f"[filter_moz1_episodes] 标记为成功的 episode 数量={len(success_eps)}，"
        f"标记为失败的 episode 数量={len(failed_eps)}"
    )
    if len(success_eps) == 0:
        print("[filter_moz1_episodes] 警告：没有找到任何成功 episode（is_mistake == False），请检查数据。")

    return success_eps


def filter_episodes_file(
    episodes_path: Path,
    output_path: Path,
    success_episode_indices: set[int],
) -> tuple[int, int]:
    """
    根据 success_episode_indices 过滤 meta/episodes.jsonl。

    episodes.jsonl 每行形如：
      {"episode_index": 0, "tasks": [...], "length": ...}
    """
    print(f"[filter_moz1_episodes] 读取 episodes: {episodes_path}")
    print(f"[filter_moz1_episodes] 写出过滤后的 episodes: {output_path}")

    total = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_path.open("r") as fin, output_path.open("w") as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            ep_index = obj.get("episode_index")
            if ep_index in success_episode_indices:
                fout.write(json.dumps(obj) + "\n")
                kept += 1

    print(
        f"[filter_moz1_episodes] 原 episodes 行数={total}，"
        f"保留成功 episode 行数={kept}"
    )
    if kept == 0:
        print("[filter_moz1_episodes] 警告：过滤后没有任何 episode，被保留的数量为 0。")

    return total, kept


def main(cfg: FilterConfig) -> None:
    dataset_root = Path(cfg.dataset_root).expanduser().absolute()
    assert dataset_root.exists(), f"dataset_root 不存在: {dataset_root}"

    event_log_path = (
        Path(cfg.event_log_path).expanduser().absolute()
        if cfg.event_log_path is not None
        else dataset_root / "event_log.jsonl"
    )
    episodes_path = (
        Path(cfg.episodes_path).expanduser().absolute()
        if cfg.episodes_path is not None
        else dataset_root / "meta" / "episodes.jsonl"
    )
    output_episodes_path = (
        Path(cfg.output_episodes_path).expanduser().absolute()
        if cfg.output_episodes_path is not None
        else dataset_root / "meta" / "episodes_success.jsonl"
    )

    assert event_log_path.exists(), f"event_log.jsonl 不存在: {event_log_path}"
    assert episodes_path.exists(), f"episodes.jsonl 不存在: {episodes_path}"

    print(f"[filter_moz1_episodes] dataset_root         = {dataset_root}")
    print(f"[filter_moz1_episodes] event_log_path       = {event_log_path}")
    print(f"[filter_moz1_episodes] episodes_path        = {episodes_path}")
    print(f"[filter_moz1_episodes] output_episodes_path = {output_episodes_path}")
    print(f"[filter_moz1_episodes] overwrite            = {cfg.overwrite}")

    success_eps = collect_success_episodes(event_log_path)
    total, kept = filter_episodes_file(
        episodes_path=episodes_path,
        output_path=output_episodes_path,
        success_episode_indices=success_eps,
    )

    if cfg.overwrite:
        backup_path = episodes_path.with_name("episodes_all_backup.jsonl")
        if not backup_path.exists():
            print(f"[filter_moz1_episodes] 备份原 episodes.jsonl -> {backup_path}")
            episodes_path.rename(backup_path)
        else:
            print(
                f"[filter_moz1_episodes] 警告：备份文件已存在，不再覆盖: {backup_path}，"
                "将直接覆盖 episodes.jsonl"
            )

        print(
            f"[filter_moz1_episodes] 使用过滤结果覆盖原 episodes.jsonl: {episodes_path}"
        )
        # 如果 output_episodes_path 就是 episodes_path，本身已经是覆盖了
        if output_episodes_path != episodes_path:
            output_episodes_path.replace(episodes_path)

    print(
        "[filter_moz1_episodes] 完成。\n"
        f"  - 原 episodes 行数: {total}\n"
        f"  - 成功 episode 行数: {kept}\n"
        f"  - 输出文件: {episodes_path if cfg.overwrite else output_episodes_path}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(FilterConfig)
    main(cfg)


