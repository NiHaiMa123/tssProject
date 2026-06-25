"""
工具模块：配置加载、日志、音频 I/O、响度归一化等
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import yaml
import numpy as np
import torch
import torchaudio
import soundfile as sf
import pyloudnorm as pyln

warnings.filterwarnings("ignore")

# ── 日志 ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chattts_pipeline")


def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)


# ── 配置 ─────────────────────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"已加载配置: {config_path}")
    return cfg


def ensure_dirs(cfg: dict):
    """确保所有输入输出目录存在"""
    for key in ["input_raw_audio", "input_texts", "input_emotion_ref",
                "output_audio", "output_speaker_emb", "temp_dir"]:
        p = Path(cfg["paths"][key])
        p.mkdir(parents=True, exist_ok=True)


# ── 设备 ─────────────────────────────────────────────────────
def get_device(device_str: str) -> torch.device:
    """获取可用的 PyTorch 设备"""
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        logger.warning(f"设备 '{device_str}' 不可用，回退至 CPU")
        return torch.device("cpu")


# ── 音频 I/O ─────────────────────────────────────────────────
def load_audio(file_path: str, target_sr: int = None) -> Tuple[torch.Tensor, int]:
    """加载音频文件，自动处理格式（wav / mp3 / m4a 等）"""
    import soundfile as sf
    file_path = str(file_path)
    try:
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim == 1:
            data = data[:, np.newaxis]
        waveform = torch.from_numpy(data.T).float()
    except Exception:
        # 回退到 torchaudio（如果安装了 torchcodec）
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            raise RuntimeError(f"无法加载音频文件: {file_path}")

    if target_sr and sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr


def save_audio(
    waveform: torch.Tensor,
    sr: int,
    file_path: str,
    normalize: bool = True,
):
    """保存音频为 wav 文件"""
    import soundfile as sf
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    wav = waveform.detach().cpu().squeeze().numpy().astype(np.float32)

    # 防止削波
    if normalize:
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak

    # soundfile 期望 (samples,) 或 (samples, channels)
    if wav.ndim == 2 and wav.shape[0] <= 2:
        wav = wav.T  # (channels, samples) -> (samples, channels)

    sf.write(str(file_path), wav, sr)
    logger.info(f"音频已保存: {file_path}")


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """多声道 → 单声道"""
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


# ── EBU R128 响度归一化 ──────────────────────────────────────
def ebu_r128_normalize(
    waveform: torch.Tensor,
    sr: int,
    target_lufs: float = -23.0,
) -> torch.Tensor:
    """EBU R128 响度归一化"""
    wav_np = waveform.squeeze().cpu().numpy().astype(np.float64)

    # 创建响度测量器
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav_np)

    if np.isinf(loudness) or np.isnan(loudness):
        logger.warning("响度测量无效，跳过归一化")
        return waveform

    # 归一化到目标响度
    wav_normalized = pyln.normalize.loudness(wav_np, loudness, target_lufs)
    result = torch.from_numpy(wav_normalized).float().unsqueeze(0)

    # 防止削波
    peak = result.abs().max()
    if peak > 0.98:
        result = result / peak * 0.98

    logger.debug(f"EBU R128 归一化: {loudness:.1f} LUFS → {target_lufs:.1f} LUFS")
    return result


# ── 文件工具 ─────────────────────────────────────────────────
def find_audio_files(directory: str) -> list:
    """查找目录下所有支持的音频文件"""
    extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
    directory = Path(directory)
    if not directory.exists():
        return []
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(files)


def find_text_files(directory: str) -> list:
    """查找目录下所有 txt 文件"""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.txt"))


def check_user_data_exists(cfg: dict) -> bool:
    """检查用户是否已放入数据"""
    raw_audio = Path(cfg["paths"]["input_raw_audio"])
    texts = Path(cfg["paths"]["input_texts"])
    has_audio = any(raw_audio.glob("*")) if raw_audio.exists() else False
    has_texts = any(texts.glob("*")) if texts.exists() else False
    return has_audio or has_texts


def cleanup_temp(temp_dir: str = "temp"):
    """清理临时文件"""
    import shutil
    temp = Path(temp_dir)
    if temp.exists():
        shutil.rmtree(temp, ignore_errors=True)
        temp.mkdir(parents=True, exist_ok=True)
    logger.info("临时文件已清理")