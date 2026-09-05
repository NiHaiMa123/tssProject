#!/usr/bin/env python3
"""
ChatTTS 零样本音色克隆与情感合成流水线 — 主入口
────────────────────────────────────────────────
用法: python main.py [--config config.yaml]
"""

import sys
import argparse
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    logger, setup_logger, load_config, ensure_dirs, setup_model_paths,
    find_audio_files, find_text_files, check_user_data_exists,
    cleanup_temp,
)
from src.audio_enhance import run_audio_enhance_pipeline
from src.synthesize import run_batch_synthesis
from src.self_test import run_self_test


def main():
    parser = argparse.ArgumentParser(
        description="ChatTTS 零样本音色克隆与情感合成流水线"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出详细日志",
    )
    parser.add_argument(
        "--skip-enhance",
        action="store_true",
        help="跳过音频增强，直接使用已有音色向量",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="跳过合成，仅执行音频增强与音色提取",
    )
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)

    # 加载配置
    cfg = load_config(args.config)

    # 统一设置模型缓存路径（必须在加载任何模型前调用）
    setup_model_paths(cfg)

    ensure_dirs(cfg)

    logger.info("=" * 60)
    logger.info("ChatTTS 零样本音色克隆与情感合成流水线")
    logger.info("=" * 60)

    # ── 自验证检测 ──
    if not check_user_data_exists(cfg):
        logger.info("未检测到用户数据，进入自验证模式...")
        success = run_self_test(cfg)
        cleanup_temp(cfg["paths"]["temp_dir"])
        sys.exit(0 if success else 1)

    # ── 模块 1：音频增强与音色提取 ──

    if not args.skip_enhance:
        audio_files = find_audio_files(cfg["paths"]["input_raw_audio"])
        if not audio_files:
            logger.error("input/raw_audio/ 中未找到音频文件")
            logger.error("支持格式: wav, mp3, m4a, flac, ogg")
            sys.exit(1)

        audio_paths = [str(f) for f in audio_files]
        logger.info(f"找到 {len(audio_paths)} 个音频文件")

        run_audio_enhance_pipeline(
            cfg=cfg,
            audio_paths=audio_paths,
            temp_dir=cfg["paths"]["temp_dir"],
            output_emb_dir=cfg["paths"]["output_speaker_emb"],
        )

    if args.skip_synthesis:
        logger.info("跳过合成阶段（--skip-synthesis）")
        cleanup_temp(cfg["paths"]["temp_dir"])
        logger.info("音频增强与参考片段提取完成!")
        return

    # ── 模块 2 & 3：文本解析与批量情感合成 ──
    text_files = find_text_files(cfg["paths"]["input_texts"])
    if not text_files:
        logger.error("input/texts/ 中未找到 .txt 文件")
        sys.exit(1)

    logger.info(f"找到 {len(text_files)} 个文本文件")

    output_files = run_batch_synthesis(
        cfg=cfg,
        output_audio_dir=cfg["paths"]["output_audio"],
    )

    # ── 清理与总结 ──
    cleanup_temp(cfg["paths"]["temp_dir"])

    logger.info("")
    logger.info("=" * 60)
    logger.info("  流水线完成!")
    logger.info("=" * 60)
    logger.info(f"  合成音频: {len(output_files)} 个文件")
    logger.info(f"  输出目录: {cfg['paths']['output_audio']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()