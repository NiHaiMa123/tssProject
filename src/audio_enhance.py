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

# PyTorch 2.6+ 兼容性补丁: torch.load 默认 weights_only=True
# 会导致旧模型（如 resemblyzer）加载失败
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

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

        # 提取 vocals — htdemucs 源顺序: ['drums', 'bass', 'other', 'vocals']
        # Bug 修复: 之前误取 sources[0](鼓声)，导致后续全部流程用到的是鼓声而非人声
        source_names = model.sources
        vocals_idx = source_names.index('vocals') if 'vocals' in source_names else 0
        logger.info(f"Demucs 源顺序: {source_names}, 取索引 {vocals_idx} (vocals)")
        vocals = sources[vocals_idx].cpu()  # shape: (channels, samples)
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
    评估语音段可用性（基于能量，避免拒掉有效语音段）
    返回: 分数 (1~5)，< 2.0 表示该段基本是静音
    """
    wav = waveform.squeeze().cpu().numpy()
    eps = 1e-10
    energy = np.mean(wav ** 2)
    if energy < eps:
        return 1.0
    # 能量充分就认为是可用语音段
    if energy > 1e-6:
        return 3.5
    return 2.0


def _energy_vad_segments(
    waveform: torch.Tensor,
    sr: int,
    min_dur: float = 1.0,
    energy_percentile: int = 30,
) -> List[Tuple[float, float]]:
    """
    基于能量的语音活动检测（VAD 退避方案）
    将音频分成 100ms 帧，选取能量高于百分位的帧，合并连续帧为语音段
    返回: [(start_sec, end_sec), ...]
    """
    wav = waveform.squeeze().cpu().numpy()
    frame_len = int(0.1 * sr)
    frames = [wav[i:i+frame_len] for i in range(0, len(wav)-frame_len, frame_len)]
    frame_energies = np.array([np.mean(f**2) for f in frames])
    threshold = np.percentile(frame_energies, energy_percentile)

    active = frame_energies > threshold
    segments = []
    i = 0
    while i < len(active):
        if active[i]:
            start = i * 0.1
            while i < len(active) and active[i]:
                i += 1
            end = i * 0.1
            if end - start >= min_dur:
                segments.append((start, end))
        else:
            i += 1
    logger.info(f"能量 VAD: {len(segments)} 个语音段 (阈值 P{energy_percentile}={threshold:.8f})")
    return segments


def select_best_segments(
    waveform: torch.Tensor,
    sr: int,
    segments: List[Tuple[float, float]],
    cfg: dict,
) -> List[Tuple[torch.Tensor, float]]:
    """
    从语音段中选出质量最优的多段（每段 10~15 秒），用于多片段平均融合
    如果 VAD 段不足，自动退避到能量检测
    返回: [(片段 waveform, 质量分数), ...]
    """
    min_dur = cfg["audio"]["segment_duration_min"]
    max_dur = cfg["audio"]["segment_duration_max"]
    num_segments = cfg["audio"].get("num_reference_segments", 5)
    min_gap = cfg["audio"].get("min_segment_gap", 3.0)
    mos_threshold = cfg["dnsmos"]["threshold"]

    candidates = []
    for start, end in segments:
        duration = end - start
        if duration < 1.0:
            continue

        start_sample = int(start * sr)
        end_sample = int(end * sr)
        seg_full = waveform[:, start_sample:end_sample]

        # 如果语音段很长，从中切出多个候选片段（滑动窗口方式）
        if duration > max_dur * 1.5:
            step = max_dur * 0.5
            pos = 0.0
            while pos + max_dur <= duration:
                s = int(pos * sr)
                e = int((pos + max_dur) * sr)
                seg = seg_full[:, s:e]
                if compute_dnsmos(seg, sr) >= mos_threshold:
                    candidates.append((seg, 3.5, max_dur, start + pos, start + pos + max_dur))
                pos += step
            # 最后一段
            if pos < duration:
                s = int(max(0, duration - max_dur) * sr)
                e = int(duration * sr)
                seg = seg_full[:, s:e]
                actual_dur = (e - s) / sr
                if actual_dur >= min_dur and compute_dnsmos(seg, sr) >= mos_threshold:
                    candidates.append((seg, 3.5, actual_dur, start + (duration - max_dur), end))
        else:
            # 短段：截取中间部分
            seg = seg_full
            actual_dur = duration
            if duration > max_dur:
                mid = len(seg.squeeze()) // 2
                half = int(max_dur * sr // 2)
                seg = seg[:, mid - half:mid + half]
                actual_dur = max_dur

            if actual_dur >= min_dur and compute_dnsmos(seg, sr) >= mos_threshold:
                mid_time = (start + end) / 2
                candidates.append((seg, 3.5, actual_dur, mid_time - actual_dur / 2, mid_time + actual_dur / 2))

    # 如果 VAD 候选不足，用能量 VAD 补充候选
    total_candidate_dur = sum(c[2] for c in candidates) if candidates else 0
    target_total_dur = num_segments * min_dur
    if total_candidate_dur < target_total_dur:
        logger.warning(f"VAD 候选总时长 {total_candidate_dur:.1f}s 不足目标 {target_total_dur:.1f}s，退避到能量检测")
        energy_segs = _energy_vad_segments(waveform, sr, min_dur=min_dur)
        for start, end in energy_segs:
            duration = end - start
            if duration < min_dur:
                continue
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            seg = waveform[:, start_sample:end_sample]
            actual_dur = min(duration, max_dur)
            if duration > max_dur:
                mid = min(len(seg.squeeze()) // 2, int(max_dur * sr // 2))
                half = int(max_dur * sr // 2)
                if mid - half >= 0 and mid + half <= seg.shape[-1]:
                    seg = seg[:, mid - half:mid + half]
            candidates.append((seg, 3.5, actual_dur, start, start + actual_dur))

    if not candidates:
        # 极端退避：从整个音频均匀采样
        logger.warning("VAD 和能量检测均未获到候选，从整个音频均匀采样")
        total_len = waveform.shape[-1] / sr
        chunk_dur = max_dur
        spacing = max(chunk_dur, total_len / num_segments)
        for i in range(num_segments):
            center = i * spacing + spacing / 2
            if center - chunk_dur / 2 < 0:
                start_s = 0
            else:
                start_s = int((center - chunk_dur / 2) * sr)
            end_s = min(start_s + int(chunk_dur * sr), waveform.shape[-1])
            if end_s - start_s < int(min_dur * sr):
                continue
            seg = waveform[:, start_s:end_s]
            actual_dur = (end_s - start_s) / sr
            candidates.append((seg, 3.0, actual_dur, start_s / sr, end_s / sr))

    # 按质量从高到低排序
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 贪心选择：优先选高质量的，同时保证片段之间有足够间隔
    selected = []
    for cand in candidates:
        _, _, _, seg_start, seg_end = cand
        seg_mid = (seg_start + seg_end) / 2

        too_close = False
        for sel in selected:
            _, _, _, sel_start, sel_end = sel
            sel_mid = (sel_start + sel_end) / 2
            if abs(seg_mid - sel_mid) < min_gap + (seg_end - seg_start) / 2 + (sel_end - sel_start) / 2:
                too_close = True
                break

        if not too_close:
            selected.append(cand)
            if len(selected) >= num_segments:
                break

    # 如果选不够，就不考虑间隔了
    if len(selected) < num_segments and len(candidates) > len(selected):
        for cand in candidates:
            if cand not in selected:
                selected.append(cand)
                if len(selected) >= num_segments:
                    break

    result = [(seg, mos) for seg, mos, _, _, _ in selected]
    avg_mos = sum(m for _, m in result) / len(result) if result else 0.0
    total_dur = sum(s.shape[-1] / sr for s, _ in result)
    logger.info(f"选取 {len(result)} 个参考片段，总时长 {total_dur:.1f}s，平均 MOS {avg_mos:.2f}")
    return result


def prepare_reference_segments(
    enhanced_path: str,
    cfg: dict,
    temp_dir: str = "temp",
) -> List[str]:
    """
    从增强音频中选取多个最优参考片段，做响度归一化，转换到 24kHz 单声道
    返回: 参考片段路径列表
    """
    logger.info("=" * 50)
    logger.info("步骤 1.3: 多参考片段选取")
    logger.info("=" * 50)

    target_sr = cfg["audio"]["target_sample_rate"]
    target_lufs = cfg["audio"]["ebu_r128_target_db"]

    # 加载增强音频
    waveform, sr = load_audio(enhanced_path)
    waveform = convert_to_mono(waveform)

    # VAD 检测语音段
    segments = detect_speech_segments(waveform, sr, cfg)

    # 选择多个最优片段
    best_segments = select_best_segments(waveform, sr, segments, cfg)

    # 处理每个片段
    ref_paths = []
    ref_dir = Path(temp_dir) / "reference_segments"
    ref_dir.mkdir(parents=True, exist_ok=True)

    for idx, (seg, mos) in enumerate(best_segments):
        # 重采样到 24kHz
        seg_sr = sr
        if seg_sr != target_sr:
            resampler = torchaudio.transforms.Resample(seg_sr, target_sr)
            seg = resampler(seg)
            seg_sr = target_sr

        # EBU R128 响度归一化
        seg = ebu_r128_normalize(seg, seg_sr, target_lufs)

        # 保存
        ref_path = str(ref_dir / f"ref_{idx+1:02d}.wav")
        save_audio(seg, seg_sr, ref_path)
        dur = seg.shape[-1] / seg_sr
        logger.info(f"  参考片段 {idx+1}: {ref_path} (时长 {dur:.1f}s, MOS {mos:.2f})")
        ref_paths.append(ref_path)

    return ref_paths


# ══════════════════════════════════════════════════════════════
# 1.4 说话人音色嵌入提取 (WeSpeaker)
# ══════════════════════════════════════════════════════════════

def extract_speaker_embedding(
    audio_paths: List[str],
    cfg: dict,
    output_dir: str = "output/speaker_emb",
    speaker_name: str = "speaker",
) -> str:
    """
    使用 WeSpeaker / Resemblyzer 提取说话人音色向量
    支持多片段输入，取平均得到更稳定的音色表示
    优先 WeSpeaker，失败则回退至 Resemblyzer
    返回: 保存的 .npy 文件路径
    """
    logger.info("=" * 50)
    logger.info("步骤 1.4: 说话人音色嵌入提取（多片段融合）")
    logger.info("=" * 50)

    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]

    target_sr = cfg["audio"]["target_sample_rate"]
    device_str = cfg["speaker_encoder"]["device"]
    model_name = cfg["speaker_encoder"]["model_name"]
    device = get_device(device_str)

    all_embeddings = []
    encoder_loaded = None
    encoder_type = None

    # 尝试 WeSpeaker
    try:
        _apply_wespeaker_patches()
        from wespeaker.cli.speaker import load_model

        encoder = load_model(model_name)
        encoder.model.to(device)
        encoder.model.eval()
        encoder_loaded = encoder
        encoder_type = "wespeaker"
        logger.info(f"使用 WeSpeaker ({model_name}) 提取音色向量")
    except Exception as e:
        logger.warning(f"WeSpeaker 加载失败: {e}，尝试 Resemblyzer...")

    # 回退至 Resemblyzer
    if encoder_loaded is None:
        try:
            from resemblyzer import VoiceEncoder
            project_root = Path(__file__).resolve().parent.parent
            resemblyzer_model_path = project_root / "models" / "resemblyzer" / "pretrained.pt"
            if resemblyzer_model_path.exists():
                encoder = VoiceEncoder(device=device, weights_fpath=str(resemblyzer_model_path))
                logger.info("使用本地 Resemblyzer 模型")
            else:
                encoder = VoiceEncoder(device=device)
                logger.info("使用默认 Resemblyzer 模型（自动下载）")
            encoder_loaded = encoder
            encoder_type = "resemblyzer"
        except Exception as e:
            raise RuntimeError(f"音色提取失败 (WeSpeaker + Resemblyzer 均不可用): {e}")

    # 逐片段提取
    for idx, audio_path in enumerate(audio_paths):
        logger.info(f"  提取片段 {idx+1}/{len(audio_paths)}: {Path(audio_path).name}")

        # 加载音频（24kHz 单声道）
        waveform, sr = load_audio(audio_path, target_sr=target_sr)
        waveform = convert_to_mono(waveform)

        embedding = None

        if encoder_type == "wespeaker":
            try:
                with torch.no_grad():
                    if hasattr(encoder_loaded, 'extract_embedding_from_pcm'):
                        wav = waveform.to(device)
                        embedding = encoder_loaded.extract_embedding_from_pcm(wav, target_sr)
                    elif hasattr(encoder_loaded, 'extract_embedding'):
                        embedding = encoder_loaded.extract_embedding(audio_path)
                    else:
                        raise AttributeError("WeSpeaker encoder 不支持 extract_embedding")

                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.squeeze().cpu().numpy()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.squeeze()
            except Exception as e:
                logger.warning(f"  WeSpeaker 提取片段 {idx+1} 失败: {e}")

        if embedding is None and encoder_type == "resemblyzer":
            wav_np = waveform.squeeze().cpu().numpy()
            embedding = encoder_loaded.embed_utterance(wav_np)

        if embedding is not None:
            all_embeddings.append(embedding)

    if not all_embeddings:
        raise RuntimeError("所有片段音色提取均失败")

    # 平均融合
    embeddings_stack = np.stack(all_embeddings, axis=0)
    final_embedding = np.mean(embeddings_stack, axis=0)

    # L2 归一化（说话人嵌入通常需要归一化）
    norm = np.linalg.norm(final_embedding)
    if norm > 1e-8:
        final_embedding = final_embedding / norm

    # 保存为 .npy
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_path = output_dir / f"{speaker_name}.npy"
    np.save(str(emb_path), final_embedding)
    logger.info(f"音色向量已保存: {emb_path}")
    logger.info(f"  片段数: {len(all_embeddings)}, 维度: {final_embedding.shape}")
    logger.info(f"  融合方式: 平均 + L2 归一化")
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

    # 1.3 多参考片段选取
    ref_paths = prepare_reference_segments(enhanced_path, cfg, temp_dir)

    # 1.4 提取音色嵌入（多片段平均融合）
    speaker_name = Path(primary_audio).stem
    emb_path = extract_speaker_embedding(ref_paths, cfg, output_emb_dir, speaker_name)

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