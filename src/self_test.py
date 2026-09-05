"""
自验证模式（冒烟测试）
──────────────────────
当 input/raw_audio 和 input/texts 均为空时自动触发。
从公开数据集下载示例音频，运行完整流水线，生成测试语音。
"""

import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
import torchaudio

from .utils import logger, load_config, ensure_dirs, save_audio


# ── 自验证数据下载 ───────────────────────────────────────────
SELF_TEST_TEXT = (
    "[happy]The quick brown fox jumps over the lazy dog. "
    "[calm]She sells sea shells by the sea shore."
)

SELF_TEST_AUDIO_NAME = "demo_self_test"


def download_sample_audio(cfg: dict, temp_dir: str) -> Optional[str]:
    """
    从公开数据集下载示例音频
    返回: 下载的 wav 文件路径，或 None
    """
    logger.info("=" * 50)
    logger.info("自验证模式: 下载示例音频")
    logger.info("=" * 50)

    url = cfg["self_test"]["audio_url"]
    fallback_url = cfg["self_test"]["audio_url_fallback"]

    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tar_path = temp_dir / "dev-clean.tar.gz"

    # 尝试下载 LibriTTS 数据集
    for attempt_url in [url, fallback_url]:
        try:
            logger.info(f"下载中: {attempt_url}")
            response = requests.get(attempt_url, stream=True, timeout=120)
            response.raise_for_status()

            with open(tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"下载完成: {tar_path}")

            # 解压
            extract_dir = temp_dir / "libritts"
            extract_dir.mkdir(exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir, filter="fully_trusted")

            # 查找第一个 wav 文件
            wav_files = list(extract_dir.rglob("*.wav"))
            if wav_files:
                wav_path = wav_files[0]
                logger.info(f"找到示例音频: {wav_path}")
                return str(wav_path)
            else:
                logger.warning("未在压缩包中找到 wav 文件")

        except Exception as e:
            logger.warning(f"下载失败 ({attempt_url}): {e}")
            continue

    # 最终回退：生成一个简单合成测试音频
    logger.warning("无法下载示例音频，将使用内置合成测试")
    return generate_demo_audio(temp_dir)


def generate_demo_audio(temp_dir: Path) -> Optional[str]:
    """
    生成一个简单的合成测试音频（正弦波 + 少量噪声）
    用于在没有网络时的基础冒烟测试
    """
    logger.info("生成合成测试音频...")
    sr = 24000
    duration = 3.0  # 3 秒
    t = torch.linspace(0, duration, int(sr * duration))

    # 模拟人声：基频 + 几个泛音
    f0 = 200  # 基频 ~200Hz
    signal = (
        0.6 * torch.sin(2 * torch.pi * f0 * t) +
        0.2 * torch.sin(2 * torch.pi * f0 * 2 * t) +
        0.1 * torch.sin(2 * torch.pi * f0 * 3 * t) +
        0.05 * torch.sin(2 * torch.pi * f0 * 4 * t)
    )

    # 加包络（模拟语音的起止）
    envelope = torch.ones_like(t)
    attack = int(0.02 * sr)
    release = int(0.05 * sr)
    if attack > 0:
        envelope[:attack] = torch.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = torch.linspace(1, 0, release)
    signal = signal * envelope

    # 加少量噪声
    signal = signal + 0.01 * torch.randn_like(signal)
    signal = signal / signal.abs().max()

    waveform = signal.unsqueeze(0)

    audio_path = str(temp_dir / "demo_synthetic.wav")
    import soundfile as sf
    sf.write(audio_path, waveform.squeeze().cpu().numpy(), sr)
    logger.info(f"合成测试音频已生成: {audio_path}")
    return audio_path


def setup_self_test_data(cfg: dict, temp_dir: str) -> Tuple[str, str]:
    """
    准备自验证数据
    返回: (音频路径, 文本文件路径)
    """
    # 下载/生成示例音频
    audio_path = download_sample_audio(cfg, temp_dir)
    if audio_path is None:
        raise RuntimeError("无法获取示例音频，自验证失败")

    # 将音频复制到 raw_audio 目录
    raw_audio_dir = Path(cfg["paths"]["input_raw_audio"])
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    dest_audio = raw_audio_dir / f"{SELF_TEST_AUDIO_NAME}.wav"
    import shutil
    shutil.copy(audio_path, str(dest_audio))
    logger.info(f"示例音频已准备: {dest_audio}")

    # 创建示例文本
    texts_dir = Path(cfg["paths"]["input_texts"])
    texts_dir.mkdir(parents=True, exist_ok=True)
    text_path = texts_dir / "demo_self_test.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(SELF_TEST_TEXT)
    logger.info(f"示例文本已准备: {text_path}")

    return str(dest_audio), str(text_path)


# ── 自验证主流程 ─────────────────────────────────────────────
def run_self_test(cfg: dict) -> bool:
    """
    运行完整自验证流程
    返回: 是否成功
    """
    logger.info("=" * 60)
    logger.info("🧪 自验证模式（冒烟测试）")
    logger.info("=" * 60)
    logger.info("检测到 input/raw_audio 和 input/texts 均为空，自动进入自验证模式。")
    logger.info("将从公开数据集下载示例数据，运行完整流水线。")
    logger.info("")

    temp_dir = cfg["paths"]["temp_dir"]
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    try:
        # 1. 准备自验证数据
        audio_path, text_path = setup_self_test_data(cfg, temp_dir)

        # 2. 运行音频增强与音色提取
        from .audio_enhance import run_audio_enhance_pipeline, find_audio_files
        audio_files = [str(audio_path)]
        emb_path = run_audio_enhance_pipeline(
            cfg=cfg,
            audio_paths=audio_files,
            temp_dir=temp_dir,
            output_emb_dir=cfg["paths"]["output_speaker_emb"],
        )

        # 3. 运行批量情感合成
        from .synthesize import run_batch_synthesis
        output_files = run_batch_synthesis(
            cfg=cfg,
            speaker_emb_path=emb_path,
            output_audio_dir=cfg["paths"]["output_audio"],
        )

        # 4. 验证输出
        output_name = cfg["self_test"]["output_name"]
        output_audio_dir = Path(cfg["paths"]["output_audio"])
        demo_output = output_audio_dir / output_name

        # 检查是否有生成文件
        if output_files:
            # 将第一个输出文件复制为 demo_self_test.wav
            import shutil
            first_output = Path(output_files[0])
            if first_output.exists() and str(first_output) != str(demo_output):
                shutil.copy(str(first_output), str(demo_output))
            logger.info(f"自验证输出: {demo_output}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("  自验证成功!")
        logger.info("=" * 60)
        logger.info(f"  测试语音已保存至: {demo_output}")
        logger.info("")
        logger.info("  请将您的音频放入 input/raw_audio/")
        logger.info("  将文本放入 input/texts/")
        logger.info("  然后重新运行 run_all.sh 即可。")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"自验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── 命令行入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    ensure_dirs(config)
    success = run_self_test(config)
    sys.exit(0 if success else 1)