#!/usr/bin/env python3
"""
为 moz1 叠衣服任务准备成功数据集。

这个脚本会：
1. 读取 episodes_success.jsonl，获取成功 episode 的索引
2. 创建新的数据集目录
3. 复制成功 episode 对应的 parquet 数据文件
4. 复制成功 episode 对应的视频文件
5. 创建新的 episodes.jsonl（只包含成功数据）
6. 复制必要的 meta 文件（modality.json, info.json, tasks.jsonl）
"""

import json
import shutil
from pathlib import Path
from typing import Set

import tyro


def load_success_episodes(episodes_success_path: Path) -> Set[int]:
    """加载成功 episode 的索引集合。"""
    success_indices = set()
    with open(episodes_success_path, "r") as f:
        for line in f:
            if line.strip():
                episode = json.loads(line)
                success_indices.add(episode["episode_index"])
    return success_indices


def copy_episode_data(
    source_data_dir: Path,
    target_data_dir: Path,
    episode_index: int,
    chunk_size: int = 1000,
):
    """复制单个 episode 的 parquet 数据文件。"""
    chunk_index = episode_index // chunk_size
    source_file = (
        source_data_dir
        / f"chunk-{chunk_index:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    target_chunk_dir = target_data_dir / f"chunk-{chunk_index:03d}"
    target_chunk_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_chunk_dir / f"episode_{episode_index:06d}.parquet"

    if source_file.exists():
        shutil.copy2(source_file, target_file)
        return True
    else:
        print(f"警告: 数据文件不存在: {source_file}")
        return False


def copy_episode_videos(
    source_videos_dir: Path,
    target_videos_dir: Path,
    episode_index: int,
    chunk_size: int = 1000,
    video_keys: list[str] = None,
):
    """复制单个 episode 的视频文件。"""
    if video_keys is None:
        video_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    chunk_index = episode_index // chunk_size
    copied_count = 0

    for video_key in video_keys:
        source_file = (
            source_videos_dir
            / f"chunk-{chunk_index:03d}"
            / video_key
            / f"episode_{episode_index:06d}.mp4"
        )
        target_video_dir = (
            target_videos_dir / f"chunk-{chunk_index:03d}" / video_key
        )
        target_video_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_video_dir / f"episode_{episode_index:06d}.mp4"

        if source_file.exists():
            shutil.copy2(source_file, target_file)
            copied_count += 1
        else:
            print(f"警告: 视频文件不存在: {source_file}")

    return copied_count


def create_new_episodes_jsonl(
    episodes_success_path: Path, target_episodes_path: Path
):
    """创建新的 episodes.jsonl，直接复制成功数据。"""
    shutil.copy2(episodes_success_path, target_episodes_path)
    print(f"创建新的 episodes.jsonl: {target_episodes_path}")


def main(
    source_dataset_root: str,
    target_dataset_root: str,
    episodes_success_filename: str = "episodes_success.jsonl",
):
    """
    准备成功数据集。

    Args:
        source_dataset_root: 源数据集根目录（snapshot 目录）
        target_dataset_root: 目标数据集根目录（新建的目录）
        episodes_success_filename: 成功 episodes 文件名
    """
    source_root = Path(source_dataset_root)
    target_root = Path(target_dataset_root)

    # 检查源目录
    if not source_root.exists():
        raise FileNotFoundError(f"源数据集目录不存在: {source_root}")

    source_meta_dir = source_root / "meta"
    source_data_dir = source_root / "data"
    source_videos_dir = source_root / "videos"

    episodes_success_path = source_meta_dir / episodes_success_filename
    if not episodes_success_path.exists():
        raise FileNotFoundError(
            f"成功 episodes 文件不存在: {episodes_success_path}"
        )

    # 创建目标目录结构
    target_root.mkdir(parents=True, exist_ok=True)
    target_meta_dir = target_root / "meta"
    target_data_dir = target_root / "data"
    target_videos_dir = target_root / "videos"
    target_meta_dir.mkdir(exist_ok=True)
    target_data_dir.mkdir(exist_ok=True)
    target_videos_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("准备 MOZ1 成功数据集")
    print("=" * 60)
    print(f"源数据集: {source_root}")
    print(f"目标数据集: {target_root}")
    print()

    # 1. 加载成功 episode 索引
    print("1. 加载成功 episode 索引...")
    success_indices = load_success_episodes(episodes_success_path)
    print(f"   成功 episode 数量: {len(success_indices)}")
    print(f"   Episode 索引: {sorted(list(success_indices))[:10]}...")
    print()

    # 2. 读取 info.json 获取配置
    info_path = source_meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json 不存在: {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)
    chunk_size = info.get("chunks_size", 1000)

    # 3. 复制 meta 文件
    print("2. 复制 meta 文件...")
    meta_files_to_copy = ["modality.json", "info.json", "tasks.jsonl"]
    for meta_file in meta_files_to_copy:
        source_file = source_meta_dir / meta_file
        if source_file.exists():
            shutil.copy2(source_file, target_meta_dir / meta_file)
            print(f"   ✓ {meta_file}")
        else:
            print(f"   警告: {meta_file} 不存在，跳过")
    
    # 3.1 处理 episodes_stats.jsonl（可选，但建议复制）
    episodes_stats_source = source_meta_dir / "episodes_stats.jsonl"
    if episodes_stats_source.exists():
        print()
        print("2.1 处理 episodes_stats.jsonl...")
        # 只复制成功 episode 的统计信息
        success_stats = []
        with open(episodes_stats_source, "r") as f:
            for line in f:
                if line.strip():
                    episode_stat = json.loads(line)
                    if episode_stat["episode_index"] in success_indices:
                        success_stats.append(line.strip())
        
        if success_stats:
            target_episodes_stats = target_meta_dir / "episodes_stats.jsonl"
            with open(target_episodes_stats, "w") as f:
                f.write("\n".join(success_stats) + "\n")
            print(f"   ✓ episodes_stats.jsonl (包含 {len(success_stats)} 个成功 episode 的统计)")
        else:
            print("   警告: 没有找到成功 episode 的统计信息")
    else:
        print("   注意: episodes_stats.jsonl 不存在，跳过（GR00T 会自动计算 stats.json）")

    # 4. 创建新的 episodes.jsonl
    print()
    print("3. 创建新的 episodes.jsonl...")
    create_new_episodes_jsonl(
        episodes_success_path, target_meta_dir / "episodes.jsonl"
    )
    print(f"   ✓ episodes.jsonl (包含 {len(success_indices)} 个成功 episode)")

    # 5. 复制数据文件
    print()
    print("4. 复制 parquet 数据文件...")
    copied_data_count = 0
    for episode_idx in sorted(success_indices):
        if copy_episode_data(source_data_dir, target_data_dir, episode_idx, chunk_size):
            copied_data_count += 1
    print(f"   ✓ 复制了 {copied_data_count}/{len(success_indices)} 个数据文件")

    # 6. 复制视频文件
    print()
    print("5. 复制视频文件...")
    video_keys = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    copied_video_count = 0
    for episode_idx in sorted(success_indices):
        count = copy_episode_videos(
            source_videos_dir,
            target_videos_dir,
            episode_idx,
            chunk_size,
            video_keys,
        )
        copied_video_count += count
    print(
        f"   ✓ 复制了 {copied_video_count}/{len(success_indices) * len(video_keys)} 个视频文件"
    )

    # 7. 更新 info.json 中的统计信息
    print()
    print("6. 更新 info.json 统计信息...")
    target_info_path = target_meta_dir / "info.json"
    with open(target_info_path, "r") as f:
        target_info = json.load(f)
    target_info["total_episodes"] = len(success_indices)
    # 计算总帧数
    total_frames = sum(
        json.loads(line)["length"]
        for line in open(target_meta_dir / "episodes.jsonl")
        if line.strip()
    )
    target_info["total_frames"] = total_frames
    target_info["splits"] = {"train": f"0:{len(success_indices)}"}

    with open(target_info_path, "w") as f:
        json.dump(target_info, f, indent=4)
    print(f"   ✓ 更新了 total_episodes={len(success_indices)}, total_frames={total_frames}")

    print()
    print("=" * 60)
    print("完成！成功数据集已准备就绪")
    print("=" * 60)
    print(f"目标数据集路径: {target_root}")
    print(f"成功 episode 数量: {len(success_indices)}")
    print()
    print("现在可以使用以下命令进行微调：")
    print(f"python scripts/gr00t_finetune.py \\")
    print(f"  --dataset-path {target_root} \\")
    print(f"  --data-config moz1_bimanual_cart \\")
    print(f"  --num-gpus 1 \\")
    print(f"  --max-steps 10000 \\")
    print(f"  --output-dir ./moz1-folding-checkpoints \\")
    print(f"  --video-backend torchcodec")


if __name__ == "__main__":
    tyro.cli(main)

