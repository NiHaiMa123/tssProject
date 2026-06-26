"""
模块 3：批量情感合成 (ChatTTS)
─────────────────────────────────
基于 ChatTTS 零样本推理，通过音色向量与情感参考音频控制生成
"""

import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torchaudio
import soundfile as sf

# PyTorch 2.6+ 兼容性补丁: torch.load 默认 weights_only=True
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from .utils import (
    logger, load_config, get_device, ensure_dirs,
    save_audio, load_audio, convert_to_mono, ebu_r128_normalize,
    find_text_files, find_audio_files,
)
from .text_parser import (
    parse_all_texts, get_emotion_params, resolve_emotion_ref_audio,
)


def _setup_hf_mirror():
    """
    配置 HuggingFace 国内镜像，加速模型下载。
    优先级：HF_ENDPOINT 环境变量 > CHATTTS_HF_MIRROR 环境变量 > config 配置
    """
    if "HF_ENDPOINT" not in os.environ:
        mirror = os.environ.get("CHATTTS_HF_MIRROR", "")
        if not mirror:
            # 尝试从 config 读取（如果已加载）
            pass
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
            logger.debug(f"使用 HuggingFace 镜像: {mirror}")


# ══════════════════════════════════════════════════════════════
# ASR 转录（SenseVoice-Small，用于 ChatTTS spk_smp 零样本克隆的 txt_smp）
# ══════════════════════════════════════════════════════════════

_sensevoice_model = None


def _get_sensevoice_model():
    """获取 SenseVoice-Small 模型单例（阿里通义 FunAudioLLM，专为中文优化）"""
    global _sensevoice_model
    if _sensevoice_model is not None:
        return _sensevoice_model
    from funasr import AutoModel
    logger.info("加载 SenseVoice-Small 模型 (中文 ASR, CPU)...")
    _sensevoice_model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        disable_update=True,
    )
    logger.info("SenseVoice-Small 模型加载完成")
    return _sensevoice_model


def transcribe_audio(wav_1d: torch.Tensor, sr: int, language: str = "zh") -> str:
    """
    用 SenseVoice-Small 转录音频为文字。
    返回纯文本（去除特殊标签），作为 ChatTTS spk_smp 的 txt_smp。

    优势（相比 faster-whisper）：
    - 中文 CER ~3%（vs faster-whisper small ~12%）
    - 原生带标点、原生简体中文输出
    - 无需重采样，SenseVoice 原生支持 16kHz
    """
    model = _get_sensevoice_model()
    # SenseVoice 要求 16kHz，重采样
    target_sr_asr = 16000
    if sr != target_sr_asr:
        wav_1d = torchaudio.functional.resample(
            wav_1d, orig_freq=sr, new_freq=target_sr_asr
        )
    # SenseVoice 通过文件路径读取，写入临时文件
    audio_np = wav_1d.detach().cpu().numpy().astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio_np, target_sr_asr)
    try:
        res = model.generate(
            input=tmp_path,
            language=language,
            use_itn=True,
        )
        # 输出格式: <|zh|><|NEUTRAL|><|Speech|><|withitn|>实际文本
        raw_text = res[0]["text"] if res else ""
        # 去除所有 <|...|> 特殊标签
        text = re.sub(r"<\|[^|]+\|>", "", raw_text).strip()
        return text
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# ChatTTS 模型加载（单例）
# ══════════════════════════════════════════════════════════════

_chattts_instance = None
_chattts_refine_params = None
_chattts_infer_params = None


def get_chattts(cfg: dict):
    """获取 ChatTTS 单例，返回 (chat, RefineTextParams, InferCodeParams)"""
    global _chattts_instance, _chattts_refine_params, _chattts_infer_params
    if _chattts_instance is not None:
        return _chattts_instance, _chattts_refine_params, _chattts_infer_params

    # 配置 HuggingFace 镜像加速下载
    hf_mirror = cfg["chattts"].get("hf_mirror", "")
    if hf_mirror and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = hf_mirror
        logger.info(f"使用 HuggingFace 镜像: {hf_mirror}")

    logger.info("加载 ChatTTS 模型...")
    try:
        import ChatTTS
        from ChatTTS import Chat
        # InferCodeParams 和 RefineTextParams 是 Chat 类的嵌套类
        RefineTextParams = Chat.RefineTextParams
        InferCodeParams = Chat.InferCodeParams
    except ImportError:
        raise ImportError("请安装 ChatTTS: pip install ChatTTS")

    device_str = cfg["chattts"]["device"]
    device = get_device(device_str)

    chat = Chat()
    chat.load(
        compile=cfg["chattts"]["compile"],
        device=device,
        source="local",
        custom_path=str(Path(cfg["paths"].get("models_dir", "models")) / "ChatTTS"),
    )

    _chattts_instance = chat
    _chattts_refine_params = RefineTextParams
    _chattts_infer_params = InferCodeParams
    logger.info(f"ChatTTS 模型加载完成 (设备: {device})")
    return _chattts_instance, _chattts_refine_params, _chattts_infer_params


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
        waveform, sr = load_audio(ref_path, target_sr=target_sr)
        waveform = convert_to_mono(waveform)
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
    spk_smp_str: Optional[str],
    spk_emb_str: Optional[str],
    emotion_ref_dir: str,
    cfg: dict,
    chat: object,
    device: torch.device,
    refine_params_cls=None,
    infer_params_cls=None,
    fallback_spk_emb: Optional[str] = None,
    txt_smp: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    合成单条文本
    参数:
        text: 待合成文本
        emotion: 情感标签
        spk_smp_str: ChatTTS speaker sample 字符串（从 sample_audio_speaker 获取）
        spk_emb_str: ChatTTS speaker embedding 字符串（从 sample_random_speaker 获取）
        emotion_ref_dir: 情感参考音频目录
        cfg: 配置
        chat: ChatTTS 实例
        device: 推理设备
        fallback_spk_emb: 固定回退音色（所有文本共用，确保音色一致）
        txt_smp: 参考音频的文字转录（spk_smp 零样本克隆必需，官方示例要求）
    返回:
        合成音频 waveform (1, samples) 或 None
    """
    if not text.strip():
        return None

    target_sr = cfg["audio"]["target_sample_rate"]
    target_lufs = cfg["audio"]["ebu_r128_target_db"]

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

    def _build_infer_kwargs(spk_emb_val, spk_smp_val, txt_smp_val=None):
        """构建推理参数（对齐官方 example.ipynb 的 Zero shot 用法）"""
        kwargs = {
            "use_decoder": cfg["chattts"].get("use_decoder", True),
            "skip_refine_text": True,
        }
        if refine_params_cls is not None and infer_params_cls is not None:
            kwargs["params_refine_text"] = refine_params_cls(
                prompt="",
                temperature=emotion_params["temperature"],
                top_P=emotion_params["top_P"],
                top_K=20,
                repetition_penalty=emotion_params["repetition_penalty"],
            )
            # 官方示例：spk_smp 克隆时不设 spk_emb，避免随机音色干扰
            # spk_smp + txt_smp(参考音频转录) 才是正确的 in-context learning 方式
            effective_spk_emb = spk_emb_val if spk_smp_val is None else None
            # 回归官方默认：min_new_token=0 + ensure_non_empty=True
            # 之前 min_new_token=100 强制禁止前100步输出EOS，导致GPT被迫
            # 生成"填充token"，音质沙哑/模糊/失真。ensure_non_empty=True
            # 仅在GPT第1步就EOS时才重新生成(官方机制)，不强制长度。
            kwargs["params_infer_code"] = infer_params_cls(
                prompt="[speed_4]",
                temperature=emotion_params["temperature"],
                top_P=emotion_params["top_P"],
                top_K=20,
                spk_emb=effective_spk_emb,
                spk_smp=spk_smp_val,
                txt_smp=txt_smp_val,
                ensure_non_empty=True,
                min_new_token=0,
            )
        return kwargs

    def _extract_wav(wavs):
        """从推理结果提取并归一化音频"""
        if wavs is None or len(wavs) == 0:
            return None
        wav = wavs[0]
        if isinstance(wav, np.ndarray):
            if wav.size == 0:
                return None
            wav = torch.from_numpy(wav).float().unsqueeze(0)
        elif isinstance(wav, list):
            if len(wav) == 0:
                return None
            wav = torch.tensor(wav).float().unsqueeze(0)
        wav = convert_to_mono(wav)
        # ── 移除 8kHz 低通滤波 ──
        # 之前用 lowpass_biquad(cutoff=8000) 完全切除8kHz以上频率，
        # 损害人声2-5kHz共振峰和8kHz+摩擦音(s/sh/f/h)，导致模糊/发闷。
        # artifact 的真正根因是 min_new_token=100 强制生成的填充token，
        # 修复 min_new_token 后不再需要高频抑制。
        wav = ebu_r128_normalize(wav, target_sr, target_lufs)
        return wav

    # ── 第 1 步: 尝试用 spk_smp (音色克隆) 推理 ──
    # 官方示例要求：spk_smp 必须配合 txt_smp（参考音频的文字转录）
    # GPT 生成有随机性，可能因 token 过短导致解码失败，重试最多 3 次
    use_spk_smp = spk_smp_str is not None and infer_params_cls is not None and txt_smp
    if use_spk_smp:
        infer_kwargs = _build_infer_kwargs(None, spk_smp_str, txt_smp)
        for attempt in range(3):
            try:
                wavs = chat.infer([text], **infer_kwargs)
                wav = _extract_wav(wavs)
                if wav is not None:
                    logger.info(f"  ✓ spk_smp 音色克隆成功 (第 {attempt+1} 次尝试)")
                    return wav
                logger.warning(f"  spk_smp 推理返回空结果 (第 {attempt+1} 次)，重试...")
            except (ValueError, RecursionError, RuntimeError) as e:
                logger.warning(f"  spk_smp 推理失败 (第 {attempt+1} 次): {e}，重试...")
            except Exception as e:
                logger.warning(f"  spk_smp 推理异常 (第 {attempt+1} 次): {e}，重试...")
        logger.warning("  spk_smp 3次重试均失败，回退到固定音色")

    # ── 第 2 步: 回退到固定音色 (fallback_spk_emb) ──
    # 使用传入的固定回退音色，确保所有文本音色一致
    if fallback_spk_emb is not None and infer_params_cls is not None:
        try:
            logger.info("  使用固定回退音色推理...")
            infer_kwargs = _build_infer_kwargs(fallback_spk_emb, None)
            wavs = chat.infer([text], **infer_kwargs)
            wav = _extract_wav(wavs)
            if wav is not None:
                logger.info("  ✓ 固定音色推理成功")
                return wav
            logger.warning("  固定音色推理返回空结果")
        except (ValueError, RecursionError, RuntimeError) as e:
            logger.warning(f"  固定音色推理失败: {e}")
        except Exception as e:
            logger.warning(f"  固定音色推理异常: {e}")

    # ── 第 3 步: 最后兜底 — 默认参数推理 ──
    try:
        logger.warning("  使用默认参数兜底推理...")
        wavs = chat.infer(
            [text],
            use_decoder=cfg["chattts"].get("use_decoder", True),
            skip_refine_text=True,
        )
        wav = _extract_wav(wavs)
        if wav is not None:
            logger.info("  ✓ 默认参数兜底成功")
            return wav
    except Exception as e2:
        logger.error(f"  兜底推理也失败: {e2}")

    logger.error(f"  ✗ 合成彻底失败: {text[:30]}...")
    return None


# ══════════════════════════════════════════════════════════════
# 批量合成
# ══════════════════════════════════════════════════════════════

def run_batch_synthesis(
    cfg: dict,
    output_audio_dir: str = "output/audio",
) -> List[str]:
    """
    批量情感合成主函数
    使用 ChatTTS 内置说话人编码器从参考音频提取音色
    返回: 生成的所有音频文件路径列表
    """
    logger.info("=" * 60)
    logger.info("模块 3: 批量情感合成")
    logger.info("=" * 60)

    target_sr = cfg["audio"]["target_sample_rate"]
    text_dir = cfg["paths"]["input_texts"]
    emotion_ref_dir = cfg["paths"]["input_emotion_ref"]
    temp_dir = Path(cfg["paths"]["temp_dir"])
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

    # ── 生成一个固定回退音色（所有文本共用，确保音色一致）──
    fallback_spk_emb = chat.sample_random_speaker()
    logger.info("已生成固定回退音色 (所有文本共用)")

    # ── 使用 ChatTTS 内置说话人编码器提取音色 ──
    # 优先使用预处理阶段生成的参考片段
    ref_seg_dir = temp_dir / "reference_segments"
    spk_smp_str = None
    spk_emb_str = None
    txt_smp = None

    if ref_seg_dir.exists():
        ref_files = sorted(ref_seg_dir.glob("ref_*.wav"))
        if ref_files:
            # ── 优化3: 遍历所有参考片段，选 RMS 最高且 ASR 能转录出有效文本的 ──
            # 历史问题: 只用 ref_files[0] 时若该片段多为静音(rms≈0.018, 静音93%)，
            # ASR 只能转录出"。"，导致 spk_smp 克隆彻底失败。
            # 修复: 按 RMS 降序逐个尝试，每个片段内选最清晰窗口 + ASR 验证。
            def _file_rms(p):
                try:
                    wf, _ = load_audio(str(p))
                    wf = convert_to_mono(wf)
                    return float((wf ** 2).mean().sqrt().item())
                except Exception:
                    return 0.0

            ref_files_by_rms = sorted(ref_files, key=_file_rms, reverse=True)
            logger.info(
                f"参考片段按 RMS 排序: "
                + ", ".join(f"{p.name}={_file_rms(p):.4f}" for p in ref_files_by_rms)
            )

            def _select_best_window(wav_1d_local, ref_dur=15.0):
                """在单个参考片段内选最清晰的 ref_dur 秒窗口
                评分 = 能量 × (1 - 频谱平坦度) × (1 - 静音占比)
                静音占比高会严重惩罚，避免选到大部分是静音的窗口"""
                max_smp = int(ref_dur * target_sr)
                if len(wav_1d_local) <= max_smp:
                    return wav_1d_local, 0.0, 0.0, 0.0, 0.0
                best_sc = -1.0
                best_st = 0
                best_e = best_f = best_sil = 0.0
                hop = max(1, max_smp // 50)
                wav_np = wav_1d_local.detach().cpu().numpy().astype(np.float32)
                n_fft = 1024
                for st in range(0, len(wav_1d_local) - max_smp + 1, hop):
                    win = wav_np[st:st + max_smp]
                    energy = float(np.mean(win ** 2))
                    if energy < 1e-6:
                        continue
                    # 静音占比 (|sample| < 0.01 视为静音)
                    sil_ratio = float(np.mean(np.abs(win) < 0.01))
                    if sil_ratio > 0.6:
                        continue  # 静音超 60% 直接跳过
                    # 频谱平坦度
                    frames = [win[i:i + n_fft] for i in range(0, len(win) - n_fft, n_fft // 2)]
                    if not frames:
                        continue
                    flats = []
                    for fr in frames:
                        spec = np.abs(np.fft.rfft(fr)) + 1e-10
                        flats.append(np.exp(np.mean(np.log(spec))) / np.mean(spec))
                    flatness = float(np.mean(flats))
                    score = energy * (1.0 - flatness) * (1.0 - sil_ratio)
                    if score > best_sc:
                        best_sc = score
                        best_st = st
                        best_e, best_f, best_sil = energy, flatness, sil_ratio
                return wav_1d_local[best_st:best_st + max_smp], best_st / target_sr, best_e, best_f, best_sil

            for ref_file in ref_files_by_rms:
                ref_path = str(ref_file)
                logger.info(f"尝试参考音频: {ref_path}")
                try:
                    waveform, _ = load_audio(ref_path)
                    waveform = convert_to_mono(waveform)
                    wav_1d = waveform.squeeze(0)

                    wav_1d_short, start_s, energy, flatness, sil = _select_best_window(wav_1d)
                    logger.info(
                        f"  选取最清晰窗口 (起始 {start_s:.1f}s, "
                        f"能量 {energy:.6f}, 平坦度 {flatness:.3f}, 静音 {sil*100:.0f}%)"
                    )

                    # 移除峰值归一化: 会改变音色特征，信任 EBU R128 响度归一化已足够

                    # ASR 转录，验证文本有效性
                    logger.info("  ASR 转录参考音频 (SenseVoice-Small)...")
                    txt_smp_try = transcribe_audio(wav_1d_short, target_sr, language="zh")
                    # 去除标点后检查是否有实际文字内容
                    txt_clean = re.sub(r"[，。！？、,.!?;:\s]+", "", txt_smp_try or "")
                    logger.info(f"  转录结果: \"{(txt_smp_try or '')[:60]}\" (有效字符 {len(txt_clean)})")

                    if len(txt_clean) < 2:
                        logger.warning(f"  该片段 ASR 转录无效(只有标点/空)，尝试下一个参考片段")
                        continue

                    # 有效：提取 spk_smp 并采用
                    spk_smp_str = chat.sample_audio_speaker(wav_1d_short)
                    spk_emb_str = None
                    txt_smp = txt_smp_try
                    logger.info(f"  ✓ ChatTTS 说话人编码提取完成 (shape: {wav_1d_short.shape})")
                    break
                except Exception as e:
                    logger.warning(f"  参考音频 {ref_path} 处理失败: {e}，尝试下一个")
                    continue
            else:
                logger.warning("所有参考片段均无法产出有效 ASR 文本，尝试直接扫描增强音频...")
                spk_smp_str = None
                txt_smp = None
        else:
            logger.warning("未找到参考片段文件，尝试直接扫描增强音频...")
            spk_smp_str = None
            txt_smp = None
    else:
        logger.warning("参考片段目录不存在，尝试直接扫描增强音频...")
        spk_smp_str = None
        txt_smp = None

    # ── 回退方案: 直接扫描增强/原始音频找有效窗口 ──
    # 当预处理生成的 ref 片段质量太差(多为静音)导致 ASR 失败时，
    # 直接在完整音频上滑动搜索一个 10s 窗口: 能量充足 + 静音<50% + ASR有效
    if spk_smp_str is None:
        candidate_sources = []
        enhanced_path = temp_dir / "enhanced.wav"
        vocals_path = temp_dir / "vocals_separated.wav"
        if enhanced_path.exists():
            candidate_sources.append(str(enhanced_path))
        if vocals_path.exists():
            candidate_sources.append(str(vocals_path))
        # 原始音频作为最后兜底
        raw_dir = Path(cfg["paths"]["input_raw_audio"])
        if raw_dir.exists():
            for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac"):
                candidate_sources.extend(str(p) for p in raw_dir.glob(ext))

        for src_path in candidate_sources:
            if spk_smp_str is not None:
                break
            logger.info(f"直接扫描音频寻找有效参考窗口: {src_path}")
            try:
                wf_full, sr_full = load_audio(src_path, target_sr=target_sr)
                wf_full = convert_to_mono(wf_full).squeeze(0)
                total_dur = len(wf_full) / target_sr
                logger.info(f"  音频总长 {total_dur:.1f}s, rms={wf_full.pow(2).mean().sqrt():.4f}")

                win_samples = int(10.0 * target_sr)
                # 从头到尾以 15s 步长扫描，找第一个有效窗口
                step_samples = int(15.0 * target_sr)
                found = False
                for start_s in range(0, max(0, len(wf_full) - win_samples), step_samples):
                    seg = wf_full[start_s:start_s + win_samples]
                    rms = float(seg.pow(2).mean().sqrt().item())
                    sil = float((seg.abs() < 0.01).float().mean().item())
                    if rms < 0.01 or sil > 0.50:
                        continue
                    logger.info(f"  候选窗口 {start_s/target_sr:.0f}s: rms={rms:.4f} silence={sil*100:.0f}%")
                    # 移除峰值归一化: 会改变音色特征
                    # ASR 验证
                    try:
                        txt_try = transcribe_audio(seg, target_sr, language="zh")
                    except Exception as ae:
                        logger.info(f"    ASR 异常: {ae}")
                        continue
                    txt_c = re.sub(r"[，。！？、,.!?;:\s]+", "", txt_try or "")
                    logger.info(f"    ASR: \"{(txt_try or '')[:50]}\" (有效字符 {len(txt_c)})")
                    if len(txt_c) >= 2:
                        spk_smp_str = chat.sample_audio_speaker(seg)
                        spk_emb_str = None
                        txt_smp = txt_try
                        logger.info(f"  ✓ 找到有效参考窗口 ({start_s/target_sr:.0f}s), spk_smp 提取成功")
                        found = True
                        break
                if not found:
                    logger.warning(f"  在 {src_path} 中未找到有效窗口")
            except Exception as e:
                logger.warning(f"  扫描 {src_path} 失败: {e}")

    if spk_smp_str is not None and txt_smp:
        logger.info("音色克隆模式: spk_smp + txt_smp (参考音频 in-context learning)")
    else:
        logger.info("音色克隆模式: 固定回退音色 (spk_emb)")

    # 逐条合成并拼接为单个音频文件
    output_files = []
    all_wavs = []  # 收集所有合成片段用于最终拼接
    success_count = 0

    for idx, (text, emotion, source_name) in enumerate(units):
        wav = synthesize_one(
            text=text,
            emotion=emotion,
            spk_smp_str=spk_smp_str,
            spk_emb_str=spk_emb_str,
            emotion_ref_dir=emotion_ref_dir,
            cfg=cfg,
            chat=chat,
            device=device,
            refine_params_cls=refine_params_cls,
            infer_params_cls=infer_params_cls,
            fallback_spk_emb=fallback_spk_emb,
            txt_smp=txt_smp,
        )

        if wav is not None:
            all_wavs.append(wav)
            success_count += 1
            logger.info(f"[{idx+1}/{len(units)}] ✓ 片段合成成功 [{emotion}]")
        else:
            logger.error(f"[{idx+1}/{len(units)}] ✗ 合成失败: {text[:30]}...")

    # 拼接所有片段为一个音频文件，中间加 0.5 秒静音间隔
    if all_wavs:
        silence_dur = 0.5
        silence = torch.zeros(1, int(silence_dur * target_sr))
        merged = []
        for i, w in enumerate(all_wavs):
            merged.append(w)
            if i < len(all_wavs) - 1:
                merged.append(silence)
        merged_wav = torch.cat(merged, dim=1)
        # EBU R128 响度归一化整体音频
        merged_wav = ebu_r128_normalize(merged_wav, target_sr, cfg["audio"]["ebu_r128_target_db"])

        source_name = units[0][2].replace(" ", "_") if units else "output"
        out_name = f"{source_name}_merged.wav"
        out_path = output_dir / out_name
        save_audio(merged_wav, target_sr, str(out_path), normalize=True)
        output_files.append(str(out_path))
        total_dur = merged_wav.shape[-1] / target_sr
        logger.info(f"拼接完成: {out_name} (总时长 {total_dur:.1f}s, {success_count}/{len(units)} 片段)")

    logger.info(f"批量合成完成: {success_count}/{len(units)} 成功")
    return output_files


# ── 命令行入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    ensure_dirs(config)
    run_batch_synthesis(config)