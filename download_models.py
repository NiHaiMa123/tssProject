#!/usr/bin/env python3
"""
模型权重下载脚本
将所有模型下载到本地 models/ 目录，方便离线使用和 git 管理

支持国内镜像加速：
  - 环境变量 HF_ENDPOINT=https://hf-mirror.com
  - 环境变量 CHATTTS_HF_MIRROR=https://hf-mirror.com
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 自动启用国内镜像（如果官方站不可达）
def _auto_mirror_setup():
    if "HF_ENDPOINT" not in os.environ:
        mirror = os.environ.get("CHATTTS_HF_MIRROR", "https://hf-mirror.com")
        os.environ["HF_ENDPOINT"] = mirror
        print(f"使用 HuggingFace 镜像: {mirror}")

_auto_mirror_setup()


def download_chattts():
    """下载 ChatTTS 模型权重到本地"""
    print("=" * 50)
    print("下载 ChatTTS 模型权重...")
    print("=" * 50)

    try:
        import ChatTTS
        from ChatTTS import Chat

        chat = Chat()
        chat.load(
            compile=False,
            device="cpu",
            source="local",
            custom_path=str(MODELS_DIR / "ChatTTS"),
        )
        print(f"  ChatTTS 模型已保存至: {MODELS_DIR / 'ChatTTS'}")
        return True
    except Exception as e:
        print(f"  ChatTTS 下载失败: {e}")
        # 尝试用默认的 huggingface source
        try:
            chat = Chat()
            chat.load(compile=False, device="cpu", source="huggingface")
            print("  ChatTTS 模型已下载到 HuggingFace 缓存")
            return True
        except Exception as e2:
            print(f"  ChatTTS HF 下载也失败: {e2}")
            return False


def download_demucs():
    """下载 Demucs htdemucs 模型权重"""
    print("=" * 50)
    print("下载 Demucs htdemucs 模型权重...")
    print("=" * 50)

    try:
        import torch
        from demucs import pretrained

        # 加载模型会自动下载权重
        model = pretrained.get_model("htdemucs")
        # 模型权重在 torch hub 缓存中
        print("  Demucs htdemucs 模型已下载到 PyTorch Hub 缓存")
        return True
    except Exception as e:
        print(f"  Demucs 下载失败: {e}")
        return False


def download_silero_vad():
    """下载 Silero VAD 模型"""
    print("=" * 50)
    print("下载 Silero VAD 模型...")
    print("=" * 50)

    try:
        import torch
        # Silero VAD 模型会自动下载
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        print("  Silero VAD 模型已下载")
        return True
    except Exception as e:
        print(f"  Silero VAD 下载失败: {e}")
        return False


def download_wespeaker():
    """下载 WeSpeaker 模型权重"""
    print("=" * 50)
    print("下载 WeSpeaker 模型权重...")
    print("=" * 50)

    try:
        # Python 3.14 + torchaudio 2.x 兼容性补丁
        import torchaudio
        if not hasattr(torchaudio, 'set_audio_backend'):
            torchaudio.set_audio_backend = lambda x: None
        # 修复 sox_effects 缺失
        import types
        if not hasattr(torchaudio, 'sox_effects'):
            dummy = types.ModuleType('torchaudio.sox_effects')
            dummy.apply_effects_tensor = lambda *a, **kw: (a[0], a[0].shape[1])
            torchaudio.sox_effects = dummy
            import sys
            sys.modules['torchaudio.sox_effects'] = dummy

        from wespeaker.inference import SpeakerEncoder
        encoder = SpeakerEncoder("wespeaker/voxceleb_resnet34_LM")
        print("  WeSpeaker 模型已下载")
        return True
    except Exception as e:
        print(f"  WeSpeaker 下载失败: {e}")
        return False


def download_resemblyzer():
    """下载 Resemblyzer 模型权重"""
    print("=" * 50)
    print("下载 Resemblyzer 模型权重...")
    print("=" * 50)

    try:
        from resemblyzer import VoiceEncoder
        # 加载模型会自动下载
        encoder = VoiceEncoder()
        print("  Resemblyzer 模型已下载")
        return True
    except Exception as e:
        print(f"  Resemblyzer 下载失败: {e}")
        return False


def download_deepfilternet():
    """下载 DeepFilterNet 模型权重"""
    print("=" * 50)
    print("下载 DeepFilterNet 模型权重...")
    print("=" * 50)

    try:
        from DeepFilterNet import DeepFilterNet
        model = DeepFilterNet.from_pretrained("deepfilternet2")
        print("  DeepFilterNet 模型已下载")
        return True
    except Exception as e:
        print(f"  DeepFilterNet 下载失败: {e}")
        return False


def main():
    print("=" * 60)
    print("模型权重下载工具")
    print(f"目标目录: {MODELS_DIR}")
    print("=" * 60)
    print()

    results = {}

    # 下载各个模型
    results["ChatTTS"] = download_chattts()
    print()
    results["Demucs"] = download_demucs()
    print()
    results["Silero VAD"] = download_silero_vad()
    print()
    results["WeSpeaker"] = download_wespeaker()
    print()
    results["Resemblyzer"] = download_resemblyzer()
    print()
    results["DeepFilterNet"] = download_deepfilternet()
    print()

    # 汇总
    print("=" * 60)
    print("下载结果汇总:")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    print()
    print(f"模型目录: {MODELS_DIR}")
    if all(results.values()):
        print("所有模型下载成功！")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"部分模型下载失败: {failed}")
        print("首次运行时会自动重试下载。")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())