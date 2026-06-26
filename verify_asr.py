#!/usr/bin/env python3
"""
ASR 验证脚本：用 SenseVoice-Small 转录合成音频，与源文本对比计算 CER
"""

import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torchaudio
import numpy as np
from src.utils import load_audio, convert_to_mono
from src.synthesize import transcribe_audio

# 源文本（与 input/texts/test.txt 一致，去除情感标签）
source_texts = [
    "你好，这是一段语音测试。",
    "今天天气真不错，适合出门散步。",
    "欢迎来到我们的系统，希望你能喜欢。",
]


def normalize_text(text: str) -> str:
    """去除标点和空白，只保留汉字，用于CER计算"""
    return re.sub(r"[，。！？、,.!?;:\s\n]+", "", text)


def compute_cer(ref: str, hyp: str) -> tuple:
    """计算字符错误率 CER = (S+D+I) / len(ref)
    返回 (cer, substitutions, deletions, insertions)"""
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if len(ref) == 0:
        return 0.0, 0, 0, len(hyp)
    # 简单编辑距离
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    ed = dp[m][n]
    cer = ed / m
    return cer, ed, m, n


def main():
    audio_path = "output/audio/test_merged.wav"
    print(f"加载合成音频: {audio_path}")
    wav, sr = load_audio(audio_path, target_sr=24000)
    wav = convert_to_mono(wav)
    wav_1d = wav.squeeze(0)
    total_dur = len(wav_1d) / sr
    print(f"总时长: {total_dur:.1f}s, 采样率: {sr}")
    print()

    # 整体转录（SenseVoice 一次转录整段）
    print("=" * 60)
    print("整体 ASR 转录（合成音频完整转录）")
    print("=" * 60)
    full_transcript = transcribe_audio(wav_1d, sr, language="zh")
    print(f"转录结果:\n{full_transcript}")
    print()

    # 拼接源文本
    full_source = "".join(source_texts)
    print(f"源文本:\n{full_source}")
    print()

    # 计算整体 CER
    cer, ed, ref_len, hyp_len = compute_cer(full_source, full_transcript)
    print(f"整体 CER: {cer*100:.1f}% (编辑距离 {ed}, 源 {ref_len} 字, 转录 {hyp_len} 字)")
    print()

    # 分段转录（按0.5秒静音切分拼接的音频，分别ASR验证每段）
    print("=" * 60)
    print("分段 ASR 转录（按静音切分，逐段验证）")
    print("=" * 60)
    # 检测静音位置切分
    wav_np = wav_1d.detach().cpu().numpy()
    frame_len = int(0.1 * sr)
    # 找到静音段（abs < 0.01）
    silence_threshold = 0.01
    min_silence_frames = 3  # 至少0.3秒静音才算分段点
    segments = []
    seg_start = 0
    silent_count = 0
    for i in range(0, len(wav_np) - frame_len, frame_len):
        frame = wav_np[i:i+frame_len]
        if np.mean(np.abs(frame)) < silence_threshold:
            silent_count += 1
        else:
            if silent_count >= min_silence_frames:
                # 静音结束，切分
                seg_end = i - silent_count * frame_len
                if seg_end - seg_start > int(0.5 * sr):  # 段长>0.5秒才保留
                    segments.append((seg_start, seg_end))
                seg_start = i
            silent_count = 0
    # 最后一段
    if len(wav_np) - seg_start > int(0.5 * sr):
        segments.append((seg_start, len(wav_np)))

    print(f"切分出 {len(segments)} 段")
    total_cer = 0
    total_ed = 0
    total_ref_len = 0
    for i, (s, e) in enumerate(segments):
        seg_wav = wav_1d[s:e]
        dur = (e - s) / sr
        seg_text = transcribe_audio(seg_wav, sr, language="zh")
        src = source_texts[i] if i < len(source_texts) else "(无源文本)"
        cer, ed, ref_len, hyp_len = compute_cer(src, seg_text)
        total_ed += ed
        total_ref_len += ref_len
        print(f"\n段{i+1} ({dur:.1f}s):")
        print(f"  源文本: {src}")
        print(f"  转录:   {seg_text}")
        print(f"  CER: {cer*100:.1f}% (编辑距离 {ed}, 源 {ref_len} 字, 转录 {hyp_len} 字)")
    if total_ref_len > 0:
        print(f"\n分段汇总 CER: {total_ed/total_ref_len*100:.1f}% (总编辑距离 {total_ed}, 总源字数 {total_ref_len})")
    print()

    # 字符级差异展示
    print("=" * 60)
    print("字符级差异（对齐展示）")
    print("=" * 60)
    ref_norm = normalize_text(full_source)
    hyp_norm = normalize_text(full_transcript)
    print(f"源文本({len(ref_norm)}字): {ref_norm}")
    print(f"转录({len(hyp_norm)}字): {hyp_norm}")
    print()

    # 标记差异位置
    diff_ref = ""
    diff_hyp = ""
    m, n = len(ref_norm), len(hyp_norm)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_norm[i-1] == hyp_norm[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    # 回溯
    i, j = m, n
    ref_marks = []
    hyp_marks = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_norm[i-1] == hyp_norm[j-1]:
            ref_marks.append(ref_norm[i-1])
            hyp_marks.append(hyp_norm[j-1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i-1][j] <= dp[i][j-1]):
            ref_marks.append(f"[删:{ref_norm[i-1]}]")
            i -= 1
        else:
            hyp_marks.append(f"[增:{hyp_norm[j-1]}]")
            j -= 1
    ref_marks.reverse()
    hyp_marks.reverse()
    print("差异标记:")
    print("源: " + "".join(ref_marks))
    print("转录: " + "".join(hyp_marks))


if __name__ == "__main__":
    main()
