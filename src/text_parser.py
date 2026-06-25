"""
模块 2：文本解析与情感映射
──────────────────────────────
解析内嵌情感标签的文本，映射到情感参考音频与控制参数
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .utils import logger, find_text_files


# ── 情感标签模式 ─────────────────────────────────────────────
EMOTION_LABELS = ["happy", "sad", "angry", "calm"]
EMOTION_PATTERN = re.compile(
    r"\[(" + "|".join(EMOTION_LABELS) + r")\]"
)

# 中文全角标点 → 半角映射（ChatTTS 对全角标点可能报警告）
_FULLWIDTH_PUNCT = str.maketrans({
    "！": "!", "？": "?", "。": ".", "，": ",", "；": ";",
    "：": ":", "（": "(", "）": ")", "＂": '"', "＇": "'",
    "｀": "`", "～": "~", "＠": "@", "＃": "#", "＄": "$",
    "％": "%", "＾": "^", "＆": "&", "＊": "*", "＿": "_",
    "＋": "+", "＝": "=", "｛": "{", "｝": "}", "［": "[",
    "］": "]", "｜": "|", "＼": "\\", "＜": "<", "＞": ">",
    "／": "/", "　": " ",
})


def _normalize_text(text: str) -> str:
    """规范化文本：转换全角标点为半角"""
    return text.translate(_FULLWIDTH_PUNCT).strip()


# ── 文本解析 ─────────────────────────────────────────────────
def parse_text_with_emotions(text: str) -> List[Tuple[str, str]]:
    """
    解析带情感标签的文本，返回 (文本片段, 情感标签) 列表
    未标记的文本归为 neutral

    示例:
        输入: "[happy]今天天气真好！[calm]我们明天再聊吧。"
        输出: [("今天天气真好！", "happy"), ("我们明天再聊吧。", "calm")]
    """
    if not text.strip():
        return []

    segments = []
    # 使用正则分割文本
    parts = EMOTION_PATTERN.split(text)

    current_emotion = "neutral"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in EMOTION_LABELS:
            current_emotion = part
        else:
            segments.append((_normalize_text(part), current_emotion))

    return segments


def parse_text_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    解析单个 .txt 文件，按行/段落拆分为合成单元
    返回: [(文本, 情感, 来源文件名), ...]
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    source_name = file_path.stem
    units = []

    # 按空行分隔段落
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        # 按行分隔
        paragraphs = [l.strip() for l in content.split("\n") if l.strip()]

    for para in paragraphs:
        # 解析情感标签
        emotion_segments = parse_text_with_emotions(para)
        for text, emotion in emotion_segments:
            units.append((text, emotion, source_name))

    return units


def parse_all_texts(text_dir: str) -> List[Tuple[str, str, str]]:
    """
    解析 input/texts/ 下所有 .txt 文件
    返回: [(文本, 情感, 来源文件名), ...]
    """
    text_dir = Path(text_dir)
    if not text_dir.exists():
        logger.warning(f"文本目录不存在: {text_dir}")
        return []

    txt_files = find_text_files(str(text_dir))
    if not txt_files:
        logger.warning(f"未找到 .txt 文件: {text_dir}")
        return []

    all_units = []
    for txt_file in txt_files:
        units = parse_text_file(str(txt_file))
        all_units.extend(units)
        logger.info(f"解析文本: {txt_file.name} → {len(units)} 个合成单元")

    return all_units


# ── 情感映射 ─────────────────────────────────────────────────
def get_emotion_params(cfg: dict, emotion: str) -> Dict[str, float]:
    """
    根据情感标签获取 ChatTTS 细粒度控制参数
    """
    emotion = emotion.lower()
    defaults = {
        "temperature": cfg["emotion"]["temperature"].get(emotion, 0.7),
        "top_P": cfg["emotion"]["top_P"].get(emotion, 0.75),
        "repetition_penalty": cfg["emotion"]["repetition_penalty"].get(emotion, 1.5),
    }
    return defaults


def resolve_emotion_ref_audio(
    emotion: str,
    emotion_ref_dir: str,
    cfg: dict,
) -> Optional[str]:
    """
    根据情感标签解析对应的参考音频路径
    若缺失则回退至 neutral.wav
    """
    emotion_ref_dir = Path(emotion_ref_dir)
    emotion = emotion.lower()

    # 查找对应情感的参考音频
    ref_path = emotion_ref_dir / f"{emotion}.wav"
    if ref_path.exists():
        return str(ref_path)

    # 尝试其他扩展名
    for ext in [".mp3", ".m4a", ".flac"]:
        alt_path = emotion_ref_dir / f"{emotion}{ext}"
        if alt_path.exists():
            return str(alt_path)

    # 回退到 neutral
    if emotion != "neutral":
        logger.debug(f"情感 '{emotion}' 参考音频缺失，回退至 neutral")
        neutral_path = emotion_ref_dir / cfg["emotion"]["neutral_ref"]
        if neutral_path.exists():
            return str(neutral_path)

    return None


# ── 命令行入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from .utils import load_config

    config = load_config()
    text_dir = config["paths"]["input_texts"]
    units = parse_all_texts(text_dir)

    for i, (text, emotion, source) in enumerate(units):
        print(f"[{i}] [{emotion}] {source}: {text[:50]}...")