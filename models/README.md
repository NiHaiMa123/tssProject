# 模型目录说明

所有模型权重文件均放在本目录下，git clone 后即可直接使用，无需再次下载。

## 目录结构

```
models/
├── ChatTTS/              # ChatTTS 主模型（GPT + DVAE + Vocos + Decoder + tokenizer）
│   ├── asset/
│   │   ├── gpt/          # GPT 模型
│   │   ├── tokenizer/    # 分词器
│   │   ├── DVAE.pt       # DVAE 模型
│   │   ├── Decoder.pt    # 解码器
│   │   ├── Vocos.pt      # Vocos 声码器
│   │   └── ...
│   └── config/           # 配置文件
├── torch_hub/            # Torch Hub 缓存（Demucs 等通过 torch.hub 加载的模型）
│   └── hub/checkpoints/
├── silero_vad/           # Silero VAD 语音活动检测模型
├── resemblyzer/          # Resemblyzer 说话人编码器（音色提取）
├── huggingface/          # HuggingFace Hub 缓存
└── deepfilternet/        # DeepFilterNet 降噪模型（可选，目前使用 noisereduce 替代）
```

## 模型大小

- **ChatTTS**: ~2.3 GB
- **Demucs (torch_hub)**: ~81 MB
- **Silero VAD**: ~11 MB
- **Resemblyzer**: ~17 MB
- **总计**: ~2.4 GB

## 注意事项

1. **Git LFS**: 所有大文件使用 Git LFS 存储，首次 clone 需要安装 git-lfs：
   ```bash
   brew install git-lfs
   git lfs install
   ```

2. **模型缺失**: 若模型文件缺失，运行下载脚本可重新下载：
   ```bash
   python download_models.py
   ```

3. **国内加速**: 下载脚本默认使用 `https://hf-mirror.com` 镜像加速，也可通过环境变量自定义：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python download_models.py
   ```

4. **离线使用**: 所有模型下载完成后，断开网络也可正常运行流水线。
