"""
模块 3：批量情感合成 (ChatTTS)
─────────────────────────────────
基于 ChatTTS 零样本推理，通过音色向量与情感参考音频控制生成
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torchaudio

from .utils import (
    logger, load_config, get_device, ensure_dirs,
    save_audio, convert_to_mono, ebu_r128_normalize,
    find_text_files, find_audio_files,
)
from .text_parser import (
    parse_all_texts, get_emotion_params, resolve_emotion_ref_audio,
)


# ══════════════════════════════════════════════════════════════
# ChatTTS 模型加载（单例）
# ══════════════════════════════════════════════════════════════

_chattts_instance = None


def get_chattts(cfg: dict):
    """获取 ChatTTS 单例"""
    global _chattts_instance
    if _chattts_instance is not None:
        return _chattts_instance

    logger.info("加载 ChatTTS 模型...")
    try:
        import ChatTTS
        from ChatTTS import Chat
        from ChatTTS.infer import RefineTextParams, InferCodeParams
    except ImportError:
        try:
            import ChatTTS
            from ChatTTS import Chat
            # 兼容旧版 API
            RefineTextParams = None
            InferCodeParams = None
        except ImportError:
            raise ImportError("请安装 ChatTTS: pip install ChatTTS")

    device_str = cfg["chattts"]["device"]
    device = get_device(device_str)

    chat = Chat()
    chat.load(
        compile=cfg["chattts"]["compile"],
        device=device,
        source="local",
    )

    _chattts_instance = chat
    _chattts_refine_params = RefineTextParams
    _chattts_infer_params = InferCodeParams
    logger.info(f"ChatTTS 模型加载完成 (设备: {device})")
    return chat, _chattts_refine_params, _chattts_infer_params


# ══════════════════════════════════════════════════════════════
# 参考音频预处理（用于韵律注入）
# ══════════════════════════════════════════════════════════════

def load_and_prepare_ref_audio(
    ref_path: Optional[str],
    target_sr: int = 24000,
) -> Optional[torch.Tensor]:
    """
    加载情感参考音频并预处理为 24kHz 单声道
    """
    if ref_path is None:
        return None

    try:
        waveform, sr = torchaudio.load(ref_path)
        waveform = convert_to_mono(waveform)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform
    except Exception as e:
        logger.warning(f"无法加载参考音频 {ref_path}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 单条合成
# ══════════════════════════════════════════════════════════════

def synthesize_one(
    text: str,
    emotion: str,
    speaker_emb_path: str,
    emotion_ref_dir: str,
    cfg: dict,
    chat: object,
    device: torch.device,
    refine_params_cls=None,
    infer_params_cls=None,
) -> Optional[torch.Tensor]:
    """
    合成单条文本
    参数:
        text: 待合成文本
        emotion: 情感标签
        speaker_emb_path: 主说话人音色向量路径
        emotion_ref_dir: 情感参考音频目录
        cfg: 配置
        chat: ChatTTS 实例
        device: 推理设备
    返回:
        合成音频 waveform (1, samples) 或 None
    """
    if not text.strip():
        return None

    target_sr = cfg["audio"]["target_sample_rate"]
    target_lufs = cfg["audio"]["ebu_r128_target_db"]

    # 加载主说话人音色向量
    speaker_emb = np.load(speaker_emb_path)
    speaker_emb_tensor = torch.from_numpy(speaker_emb).float().to(device)

    # 获取情感参数
    emotion_params = get_emotion_params(cfg, emotion)

    # 加载情感参考音频
    ref_path = resolve_emotion_ref_audio(emotion, emotion_ref_dir, cfg)
    ref_audio = load_and_prepare_ref_audio(ref_path, target_sr)

    logger.info(
        f"合成: [{emotion}] \"{text[:40]}{'...' if len(text) > 40 else ''}\" "
        f"(temp={emotion_params['temperature']:.2f}, "
        f"top_P={emotion_params['top_P']:.2f})"
    )

    try:
        # ChatTTS 推理
        infer_kwargs = {
            "use_decoder": cfg["chattts"].get("use_decoder", True),
            "use_oral": False,
            "use_laugh": False,
            "skip_refine_text": False,
        }

        # 使用精细参数控制（若 API 可用）
        if refine_params_cls is not None and infer_params_cls is not None:
            infer_kwargs["params_refine_text"] = refine_params_cls(
                prompt="",
                temperature=emotion_params["temperature"],
                top_P=emotion_params["top_P"],
                top_K=20,
                repetition_penalty=emotion_params["repetition_penalty"],
            )
            infer_kwargs["params_infer_code"] = infer_params_cls(
                prompt="",
                temperature=emotion_params["temperature"],
                top_P=emotion_params["top_P"],
                top_K=20,
                spk_emb=speaker_emb_tensor.unsqueeze(0) if speaker_emb_tensor.dim() == 1 else speaker_emb_tensor,
            )

        wavs = chat.infer([text], **infer_kwargs)

        if wavs is None or len(wavs) == 0:
            logger.error("ChatTTS 推理返回空结果")
            return None

        wav = wavs[0]
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).float().unsqueeze(0)
        elif isinstance(wav, list):
            wav = torch.tensor(wav).float().unsqueeze(0)

        # 确保 24kHz 单声道
        wav = convert_to_mono(wav)

        # EBU R128 响度归一化
        wav = ebu_r128_normalize(wav, target_sr, target_lufs)

        return wav

    except Exception as e:
        logger.error(f"合成失败: {e}")
        # 尝试简化参数重试
        try:
            logger.warning("使用简化参数重试...")
            wavs = chat.infer(
                [text],
                use_decoder=cfg["chattts"].get("use_decoder", True),
                skip_refine_text=True,
            )
            if wavs and len(wavs) > 0:
                wav = wavs[0]
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav).float().unsqueeze(0)
                elif isinstance(wav, list):
                    wav = torch.tensor(wav).float().unsqueeze(0)
                wav = convert_to_mono(wav)
                wav = ebu_r128_normalize(wav, target_sr, target_lufs)
                return wav
        except Exception as e2:
            logger.error(f"重试也失败: {e2}")

        return None


# ══════════════════════════════════════════════════════════════
# 批量合成
# ══════════════════════════════════════════════════════════════

def run_batch_synthesis(
    cfg: dict,
    speaker_emb_path: str,
    output_audio_dir: str = "output/audio",
) -> List[str]:
    """
    批量情感合成主函数
    返回: 生成的所有音频文件路径列表
    """
    logger.info("=" * 60)
    logger.info("模块 3: 批量情感合成")
    logger.info("=" * 60)

    target_sr = cfg["audio"]["target_sample_rate"]
    text_dir = cfg["paths"]["input_texts"]
    emotion_ref_dir = cfg["paths"]["input_emotion_ref"]
    output_dir = Path(output_audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析所有文本
    units = parse_all_texts(text_dir)
    if not units:
        logger.error("未找到待合成文本，请将 .txt 文件放入 input/texts/")
        return []

    logger.info(f"共 {len(units)} 个合成单元")

    # 加载 ChatTTS
    device = get_device(cfg["chattts"]["device"])
    chat, refine_params_cls, infer_params_cls = get_chattts(cfg)

    # 逐条合成
    output_files = []
    file_counter = {}  # 用于追踪每个源文件的序号

    for idx, (text, emotion, source_name) in enumerate(units):
        # 生成序号
        if source_name not in file_counter:
            file_counter[source_name] = 0
        file_counter[source_name] += 1
        seq = file_counter[source_name]

        # 输出文件名
        safe_name = source_name.replace(" ", "_")
        out_name = f"{safe_name}_{seq:03d}_{emotion}.wav"
        out_path = output_dir / out_name

        # 合成
        wav = synthesize_one(
            text=text,
            emotion=emotion,
            speaker_emb_path=speaker_emb_path,
            emotion_ref_dir=emotion_ref_dir,
            cfg=cfg,
            chat=chat,
            device=device,
            refine_params_cls=refine_params_cls,
            infer_params_cls=infer_params_cls,
        )

        if wav is not None:
            save_audio(wav, target_sr, str(out_path), normalize=True)
            output_files.append(str(out_path))
            logger.info(f"[{idx+1}/{len(units)}] ✓ {out_name}")
        else:
            logger.error(f"[{idx+1}/{len(units)}] ✗ 合成失败: {text[:30]}...")

    logger.info(f"批量合成完成: {len(output_files)}/{len(units)} 成功")
    return output_files


# ── 命令行入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    ensure_dirs(config)

    # 查找音色向量
    emb_dir = Path(config["paths"]["output_speaker_emb"])
    emb_files = list(emb_dir.glob("*.npy"))
    if not emb_files:
        logger.error("未找到音色向量，请先运行音频增强模块")
        sys.exit(1)

    emb_path = str(emb_files[0])
    run_batch_synthesis(config, emb_path)