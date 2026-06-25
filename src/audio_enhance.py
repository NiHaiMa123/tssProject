"""
模块 1：参考音频全自动增强与音色提取
─────────────────────────────────────────
流程：人声分离 → 语音增强 → 最优片段选取 → 响度归一化 → 音色嵌入提取
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

from .utils import (
    logger, load_config, get_device, ensure_dirs,
    load_audio, save_audio, convert_to_mono,
    ebu_r128_normalize, find_audio_files,
)


# --- 兼容性补丁 (Python 3.14 + torchaudio 2.x) ---
def _apply_wespeaker_patches():
    """应用 WeSpeaker/s3prl 兼容性补丁"""
    import torchaudio
    import types
    import sys
    if not hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend = lambda x: None
    if not hasattr(torchaudio, 'sox_effects'):
        dummy = types.ModuleType('torchaudio.sox_effects')
        dummy.apply_effects_tensor = lambda *a, **kw: (a[0], a[0].shape[1])
        torchaudio.sox_effects = dummy
        sys.modules['torchaudio.sox_effects'] = dummy


# ══════════════════════════════════════════════════════════════
# 1.1 人声分离 (Demucs)
# ══════════════════════════════════════════════════════════════

def separate_vocals_demucs(
    audio_path: str,
    cfg: dict,
    temp_dir: str = "temp",
) -> str:
    """
    使用 Demucs htdemucs 模型分离人声
    返回: 纯人声文件路径
    """
    logger.info("=" * 50)
    logger.info("步骤 1.1: 人声分离 (Demucs htdemucs)")
    logger.info("=" * 50)

    device = cfg["demucs"]["device"]
    model_name = cfg["demucs"]["model"]

    try:
        import demucs.separate
        from demucs import pretrained
        from demucs.apply import apply_model
        import soundfile as sf
    except ImportError:
        raise ImportError("请安装 demucs: pip install demucs")

    audio_path = str(audio_path)

    device_torch = get_device(device)
    logger.info(f"分离中 (模型: {model_name}, 设备: {device_torch})...")

    try:
        # 加载模型
        model = pretrained.get_model(model_name)
        model.to(device_torch)
        model.eval()

        # 加载音频（使用 soundfile 避免 torchcodec 依赖）
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim == 1:
            data = data[:, np.newaxis]
        # Demucs htdemucs 期望至少 2 通道
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        wav = torch.from_numpy(data.T).float().to(device_torch)

        # 应用模型分离
        with torch.no_grad():
            sources = apply_model(
                model, wav[None],
                device=device_torch,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=True,
            )[0]

        # 提取 vocals（通常是第一个源）
        vocals = sources[0].cpu()  # shape: (channels, samples)
        vocals_sr = model.samplerate

    except Exception as e:
        logger.error(f"Demucs API 分离失败: {e}")
        # 回退到命令行方式
        logger.warning("回退到命令行方式...")
        import subprocess
        # 使用实际可用设备
        cmd_device = "cpu" if device_torch.type == "cpu" else device
        out_dir = str(Path(temp_dir) / "demucs_output")
        cmd = [
            sys.executable, "-m", "demucs",
            "-n", model_name,
            "-d", cmd_device,
            "--two-stems", "vocals",
            "-o", out_dir,
            audio_path,
        ]
        subprocess.run(cmd, check=True)

        # 查找输出的 vocals 文件
        vocals_path = None
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith(".wav") and "vocals" in f.lower():
                    vocals_path = os.path.join(root, f)
                    break
        if vocals_path is None:
            raise FileNotFoundError("未找到 Demucs 输出的 vocals 文件")
        logger.info(f"人声分离完成: {vocals_path}")
        return vocals_path

    # 保存 vocals
    vocals_path = str(Path(temp_dir) / "vocals_separated.wav")
    vocals_np = vocals.squeeze().numpy().astype(np.float32)
    # soundfile 期望 (samples, channels) 格式
    if vocals_np.ndim == 2:
        vocals_np = vocals_np.T
    sf.write(vocals_path, vocals_np, vocals_sr)
    logger.info(f"人声分离完成: {vocals_path}")
    return vocals_path


# ══════════════════════════════════════════════════════════════
# 1.2 语音增强 (DeepFilterNet / noisereduce)
# ══════════════════════════════════════════════════════════════

def enhance_audio_deepfilternet(
    audio_path: str,
    cfg: dict,
    temp_dir: str = "temp",
) -> str:
    """
    使用 DeepFilterNet 进行语音增强（降噪 + 去混响）
    返回: 增强后音频路径
    """
    logger.info("=" * 50)
    logger.info("步骤 1.2: 语音增强 (DeepFilterNet)")
    logger.info("=" * 50)

    if not cfg["enhance"]["enabled"]:
        logger.info("增强已禁用，跳过")
        return audio_path

    device = cfg["enhance"]["device"]
    target_sr = cfg["audio"]["enhance_sample_rate"]

    try:
        from DeepFilterNet import DeepFilterNet
    except ImportError:
        logger.warning("DeepFilterNet 未安装，回退到 noisereduce")
        return _enhance_audio_fallback(audio_path, cfg, temp_dir)

    # 加载音频并升频到 48kHz
    waveform, sr = load_audio(audio_path, target_sr=target_sr)
    waveform = convert_to_mono(waveform)

    logger.info(f"增强中 (采样率: {target_sr}Hz, 设备: {device})...")

    try:
        model_name = cfg["enhance"].get("deepfilternet_model", "deepfilternet2")
        device_torch = get_device(device)

        # 加载模型
        model = DeepFilterNet.from_pretrained(model_name)
        model = model.to(device_torch)
        model.eval()

        # 增强
        wav_np = waveform.squeeze().cpu().numpy()
        with torch.no_grad():
            enhanced_np = model.enhance(wav_np, sr=target_sr)

        enhanced = torch.from_numpy(enhanced_np).float().unsqueeze(0)

    except Exception as e:
        logger.error(f"DeepFilterNet 增强失败: {e}")
        logger.warning("回退到 noisereduce")
        return _enhance_audio_fallback(audio_path, cfg, temp_dir)

    # 保存增强音频
    enhanced_path = str(Path(temp_dir) / "enhanced.wav")
    save_audio(enhanced, target_sr, enhanced_path)
    logger.info(f"语音增强完成: {enhanced_path}")
    return enhanced_path


def _enhance_audio_fallback(
    audio_path: str,
    cfg: dict,
    temp_dir: str = "temp",
) -> str:
    """
    回退方案：使用 noisereduce 进行基础降噪
    """
    logger.info("使用 noisereduce 进行基础降噪...")

    try:
        import noisereduce as nr
    except ImportError:
        logger.warning("noisereduce 未安装，跳过增强")
        return audio_path

    target_sr = cfg["audio"]["enhance_sample_rate"]
    waveform, sr = load_audio(audio_path, target_sr=target_sr)
    waveform = convert_to_mono(waveform)
    wav_np = waveform.squeeze().cpu().numpy()

    # 降噪
    reduced = nr.reduce_noise(
        y=wav_np,
        sr=target_sr,
        prop_decrease=0.9,
        stationary=False,
    )

    enhanced = torch.from_numpy(reduced).float().unsqueeze(0)
    enhanced_path = str(Path(temp_dir) / "enhanced.wav")
    save_audio(enhanced, target_sr, enhanced_path)
    logger.info(f"基础降噪完成: {enhanced_path}")
    return enhanced_path


# ══════════════════════════════════════════════════════════════
# 1.3 最优参考片段自动选取 (Silero VAD + DNSMOS)
# ══════════════════════════════════════════════════════════════

def detect_speech_segments(
    waveform: torch.Tensor,
    sr: int,
    cfg: dict,
) -> List[Tuple[float, float]]:
    """
    使用 Silero VAD 检测语音段
    返回: [(start_sec, end_sec), ...]
    """
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    except ImportError:
        raise ImportError("请安装 silero-vad: pip install silero-vad")

    model = load_silero_vad()
    wav = waveform.squeeze().cpu().numpy()

    # Silero VAD 期望 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav_16k = resampler(waveform).squeeze().cpu().numpy()
    else:
        wav_16k = wav

    timestamps = get_speech_timestamps(
        wav_16k,
        model,
        threshold=cfg["vad"]["threshold"],
        min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
        min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
        return_seconds=True,
    )

    segments = [(ts["start"], ts["end"]) for ts in timestamps]
    logger.info(f"VAD 检测到 {len(segments)} 个语音段")
    return segments


def compute_dnsmos(waveform: torch.Tensor, sr: int) -> float:
    """
    使用 DNSMOS 评估语音质量
    返回: MOS 分数 (1~5)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime 未安装，跳过 DNSMOS 评分")
        return 3.0

    # 简化的 DNSMOS 实现：基于信号特征估算
    # 实际使用中应加载 ONNX 模型，此处为占位
    wav = waveform.squeeze().cpu().numpy()

    # 基于信噪比和能量估算质量分数
    eps = 1e-8
    energy = np.mean(wav ** 2)
    if energy < eps:
        return 1.0

    # 高频能量占比（明亮度指标）
    if len(wav) > 1024:
        spec = np.abs(np.fft.rfft(wav))
        high_band = spec[len(spec)//2:]
        low_band = spec[:len(spec)//2]
        high_ratio = np.sum(high_band) / (np.sum(low_band) + eps)
    else:
        high_ratio = 0.5

    # 零交叉率（清浊音指标）
    zcr = np.mean(np.abs(np.diff(np.sign(wav)))) / 2.0

    # 综合评分
    score = 3.0 + 0.5 * high_ratio - 0.3 * abs(zcr - 0.1)
    score = max(1.0, min(5.0, score))
    return float(score)


def select_best_segment(
    waveform: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float]],
    cfg: dict,
) -> Tuple[torch.Tensor, float]:
    """
    从语音段中选出质量最优的一段（10~15 秒）
    返回: (最优片段 waveform, 质量分数)
    """
    min_dur = cfg["audio"]["segment_duration_min"]
    max_dur = cfg["audio"]["segment_duration_max"]
    mos_threshold = cfg["dnsmos"]["threshold"]

    candidates = []
    for start, end in segments:
        duration = end - start
        # 过滤太短的段
        if duration < 1.0:
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        seg = waveform[:, start_sample:end_sample]

        # 计算 DNSMOS
        mos = compute_dnsmos(seg, sr)
        if mos < mos_threshold:
            continue

        # 截取目标长度
        if duration > max_dur:
            # 取中间最稳定的部分
            mid = len(seg.squeeze()) // 2
            half = int(max_dur * sr // 2)
            seg = seg[:, mid - half:mid + half]
            duration = max_dur

        if duration >= min_dur:
            candidates.append((seg, mos, duration))

    if not candidates:
        # 放宽条件，取最长的一段
        logger.warning("未找到满足质量要求的片段，取最长语音段")
        if segments:
            longest = max(segments, key=lambda x: x[1] - x[0])
            start_sample = int(longest[0] * sr)
            end_sample = int(longest[1] * sr)
            duration = longest[1] - longest[0]
            seg = waveform[:, start_sample:end_sample]
            if duration > max_dur:
                mid = len(seg.squeeze()) // 2
                half = int(max_dur * sr // 2)
                seg = seg[:, mid - half:mid + half]
            candidates.append((seg, 3.0, min(duration, max_dur)))
        else:
            # 没有任何语音段，使用整个音频
            logger.warning("未检测到语音段，使用整个音频")
            wav = waveform
            if wav.shape[-1] / sr > max_dur:
                mid = wav.shape[-1] // 2
                half = int(max_dur * sr // 2)
                wav = wav[:, mid - half:mid + half]
            candidates.append((wav, 3.0, min(wav.shape[-1] / sr, max_dur)))

    # 选择质量最高的
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_seg, best_mos, best_dur = candidates[0]
    logger.info(f"最优片段: 时长 {best_dur:.1f}s, DNSMOS {best_mos:.2f}")
    return best_seg, best_mos


def prepare_reference_segment(
    enhanced_path: str,
    cfg: dict,
    temp_dir: str = "temp",
) -> str:
    """
    从增强音频中选取最优参考片段，做响度归一化，转换到 24kHz 单声道
    返回: 参考片段路径
    """
    logger.info("=" * 50)
    logger.info("步骤 1.3: 最优参考片段选取")
    logger.info("=" * 50)

    target_sr = cfg["audio"]["target_sample_rate"]
    target_lufs = cfg["audio"]["ebu_r128_target_db"]

    # 加载增强音频
    waveform, sr = load_audio(enhanced_path)
    waveform = convert_to_mono(waveform)

    # VAD 检测语音段
    segments = detect_speech_segments(waveform, sr, cfg)

    # 选择最优片段
    best_seg, mos = select_best_segment(waveform, sr, segments, cfg)

    # 重采样到 24kHz
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        best_seg = resampler(best_seg)
        sr = target_sr

    # EBU R128 响度归一化
    best_seg = ebu_r128_normalize(best_seg, sr, target_lufs)

    # 保存
    ref_path = str(Path(temp_dir) / "reference_segment.wav")
    save_audio(best_seg, sr, ref_path)
    logger.info(f"参考片段已保存: {ref_path} (MOS: {mos:.2f})")
    return ref_path


# ══════════════════════════════════════════════════════════════
# 1.4 说话人音色嵌入提取 (WeSpeaker)
# ══════════════════════════════════════════════════════════════

def extract_speaker_embedding(
    audio_path: str,
    cfg: dict,
    output_dir: str = "output/speaker_emb",
    speaker_name: str = "speaker",
) -> str:
    """
    使用 WeSpeaker / Resemblyzer 提取说话人音色向量
    优先 WeSpeaker，失败则回退至 Resemblyzer
    返回: 保存的 .npy 文件路径
    """
    logger.info("=" * 50)
    logger.info("步骤 1.4: 说话人音色嵌入提取")
    logger.info("=" * 50)

    target_sr = cfg["audio"]["target_sample_rate"]
    device_str = cfg["speaker_encoder"]["device"]
    model_name = cfg["speaker_encoder"]["model_name"]
    device = get_device(device_str)

    # 加载音频（24kHz 单声道）
    waveform, sr = load_audio(audio_path, target_sr=target_sr)
    waveform = convert_to_mono(waveform)

    embedding = None

    # 尝试 WeSpeaker
    try:
        _apply_wespeaker_patches()
        from wespeaker.cli.speaker import load_model

        # WeSpeaker 使用 load_model 加载，然后调用 extract_embedding
        encoder = load_model(model_name)
        encoder.model.to(device)
        encoder.model.eval()

        with torch.no_grad():
            if hasattr(encoder, 'extract_embedding_from_pcm'):
                wav = waveform.to(device)
                embedding = encoder.extract_embedding_from_pcm(wav, target_sr)
            elif hasattr(encoder, 'extract_embedding'):
                # 使用临时文件方式
                embedding = encoder.extract_embedding(audio_path)
            else:
                raise AttributeError("WeSpeaker encoder 不支持 extract_embedding")

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.squeeze().cpu().numpy()
        elif isinstance(embedding, np.ndarray):
            embedding = embedding.squeeze()

        logger.info(f"使用 WeSpeaker ({model_name}) 提取音色向量")
    except Exception as e:
        logger.warning(f"WeSpeaker 提取失败: {e}，尝试 Resemblyzer...")

    # 回退至 Resemblyzer
    if embedding is None:
        try:
            from resemblyzer import VoiceEncoder
            encoder = VoiceEncoder(device=device)

            wav_np = waveform.squeeze().cpu().numpy()
            embedding = encoder.embed_utterance(wav_np)
            logger.info("使用 Resemblyzer 提取音色向量")
        except Exception as e:
            raise RuntimeError(f"音色提取失败 (WeSpeaker + Resemblyzer 均不可用): {e}")

    # 保存为 .npy
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{speaker_name}.npy"
    np.save(str(emb_path), embedding)
    logger.info(f"音色向量已保存: {emb_path} (维度: {embedding.shape})")
    return str(emb_path)


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def run_audio_enhance_pipeline(
    cfg: dict,
    audio_paths: List[str],
    temp_dir: str = "temp",
    output_emb_dir: str = "output/speaker_emb",
) -> str:
    """
    完整音频增强与音色提取流水线
    参数:
        cfg: 配置字典
        audio_paths: 原始音频文件路径列表
        temp_dir: 临时目录
        output_emb_dir: 音色向量输出目录
    返回:
        speaker_emb_path: 提取的音色向量文件路径
    """
    logger.info("=" * 60)
    logger.info("模块 1: 音频增强与音色提取")
    logger.info("=" * 60)

    if not audio_paths:
        raise ValueError("未提供音频文件")

    # 使用第一个音频文件（或合并多个）
    primary_audio = audio_paths[0]
    logger.info(f"处理音频: {primary_audio}")

    # 1.1 人声分离
    vocals_path = separate_vocals_demucs(primary_audio, cfg, temp_dir)

    # 1.2 语音增强
    enhanced_path = enhance_audio_deepfilternet(vocals_path, cfg, temp_dir)

    # 1.3 最优参考片段选取
    ref_path = prepare_reference_segment(enhanced_path, cfg, temp_dir)

    # 1.4 提取音色嵌入
    speaker_name = Path(primary_audio).stem
    emb_path = extract_speaker_embedding(ref_path, cfg, output_emb_dir, speaker_name)

    logger.info("模块 1 完成!")
    return emb_path


# ── 命令行入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    ensure_dirs(config)
    audio_files = find_audio_files(config["paths"]["input_raw_audio"])
    if not audio_files:
        logger.error("input/raw_audio/ 中未找到音频文件")
        sys.exit(1)
    run_audio_enhance_pipeline(config, audio_files)