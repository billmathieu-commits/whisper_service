# Speech Service

基于 FastAPI 的语音服务，集成了 OpenAI Whisper 的语音转文本（STT）和 Microsoft Edge TTS 的文字转语音（TTS），支持 GPU 加速，使用 Docker 部署。

## 特性

### STT (语音转文本)
- 基于 OpenAI Whisper 模型
- 支持 NVIDIA GPU 加速（RTX 5090D 等显卡）
- 支持多语言转录和翻译
- 支持多种音频格式（wav, mp3, m4a, ogg, flac 等）

### TTS (文字转语音)
- 基于 Microsoft Edge TTS（edge-tts）
- 无需 API key，完全免费
- 支持多种语言和声音
- 可调节语速、音量、音调
- 高质量音频输出

### 通用特性
- Docker 容器化部署
- RESTful API 接口
- 完整的 API 文档（Swagger UI）

## 系统要求

### 硬件要求
- NVIDIA GPU（推荐 RTX 5090D 或其他 CUDA 兼容显卡）
- 至少 8GB GPU 显存（使用 base 模型）
- 更大模型需要更多显存

### 软件要求
- Docker
- Docker Compose
- NVIDIA Container Toolkit（用于 GPU 支持）
- NVIDIA 驱动版本 >= 525.60.13

## 安装 NVIDIA Container Toolkit

在主机上安装 NVIDIA Container Toolkit 以支持 Docker GPU 访问：

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/billmathieu-commits/whisper_service.git
cd whisper_service
```

### 2. 配置模型大小

编辑 `docker-compose.yml`，修改 `MODEL_SIZE` 环境变量：

```yaml
environment:
  - MODEL_SIZE=base  # 可选: tiny, base, small, medium, large
```

**模型大小对比：**
- `tiny`: 最快，准确度最低（~1GB 显存）
- `base`: 平衡选择（~1.5GB 显存）
- `small`: 更高准确度（~2GB 显存）
- `medium`: 高准确度（~5GB 显存）
- `large`: 最高准确度（~10GB 显存）

### 3. 构建并启动服务

```bash
docker-compose up --build
```

后台运行：
```bash
docker-compose up -d --build
```

### 4. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 查看服务文档
浏览器打开: http://localhost:8000/docs
```

## API 使用

### 健康检查

```bash
curl http://localhost:8000/health
```

响应示例：
```json
{
  "status": "healthy",
  "model_size": "base",
  "device": "cuda",
  "cuda_available": true
}
```

### 转录音频文件

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio.wav" \
  -F "language=zh"
```

**参数说明：**
- `file`: 音频文件（必需）
- `language`: 语言代码（可选，如 `zh`, `en`, `ja`，不指定则自动检测）
- `task`: 任务类型（可选，`transcribe` 或 `translate`，默认 `transcribe`）

响应示例：
```json
{
  "text": "这是一段测试音频",
  "language": "zh",
  "duration": 5.2
}
```

### Python 客户端示例

使用提供的测试脚本：

```bash
# 测试健康检查
python test_client.py --health

# 转录音频文件（自动检测语言）
python test_client.py --file audio.wav

# 转录音频文件（指定语言）
python test_client.py --file audio.mp3 --language zh
```

### Python 代码示例

```python
import requests

# 转录音频
with open("audio.wav", "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    data = {"language": "zh"}  # 可选
    response = requests.post("http://localhost:8000/transcribe", files=files, data=data)

result = response.json()
print(result["text"])
```

### JavaScript 示例

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('language', 'zh');

fetch('http://localhost:8000/transcribe', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data.text));
```

### TTS (文字转语音)

#### 列出可用声音

```bash
# 列出所有声音
curl "http://localhost:8000/tts/voices"

# 按语言筛选
curl "http://localhost:8000/tts/voices?language=zh-CN"
```

#### 合成语音

```bash
curl -X POST "http://localhost:8000/tts/synthesize" \
  -F "text=你好，世界！" \
  -F "voice=zh-CN-XiaoxiaoNeural" \
  --output hello.mp3
```

**参数说明：**
- `text`: 要转换的文本（必需，最多 5000 字符）
- `voice`: TTS 声音名称（可选，默认 `zh-CN-XiaoxiaoNeural`）
- `rate`: 语速调整（可选，如 `+10%`, `-20%`，默认 `+0%`）
- `volume`: 音量调整（可选，如 `+10%`, `-20%`，默认 `+0%`）
- `pitch`: 音调调整（可选，如 `+50Hz`, `-50Hz`，默认 `+0Hz`）

**常用声音：**
- 中文女声: `zh-CN-XiaoxiaoNeural`
- 中文男声: `zh-CN-YunyangNeural`
- 英文女声: `en-US-JennyNeural`
- 英文男声: `en-US-GuyNeural`
- 日文女声: `ja-JP-NanamiNeural`

使用 `/tts/voices` 接口查看所有可用声音。

### TTS Python 客户端示例

使用提供的 TTS 测试脚本：

```bash
# 列出所有声音
python test_tts_client.py --list-voices

# 列出中文声音
python test_tts_client.py --list-voices --language zh-CN

# 合成语音（默认参数）
python test_tts_client.py --text "你好，世界！"

# 合成语音（自定义声音和参数）
python test_tts_client.py --text "Hello, world!" \
  --voice en-US-JennyNeural \
  --rate=+20% \
  --volume=+10% \
  --output hello.mp3
```

### TTS Python 代码示例

```python
import requests

# 合成语音
data = {
    "text": "你好，世界！",
    "voice": "zh-CN-XiaoxiaoNeural",
    "rate": "+10%",
    "volume": "+5%",
    "pitch": "+0Hz"
}

response = requests.post("http://localhost:8000/tts/synthesize", data=data)

if response.status_code == 200:
    with open("output.mp3", "wb") as f:
        f.write(response.content)
    print("语音合成成功！")
else:
    print(f"错误: {response.text}")
```

### TTS JavaScript 示例

```javascript
const formData = new FormData();
formData.append('text', '你好，世界！');
formData.append('voice', 'zh-CN-XiaoxiaoNeural');
formData.append('rate', '+10%');

fetch('http://localhost:8000/tts/synthesize', {
  method: 'POST',
  body: formData
})
.then(response => response.blob())
.then(blob => {
  // 创建下载链接
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'output.mp3';
  a.click();
});
```

## 性能优化

### GPU 加速

服务会自动检测并使用可用的 GPU。检查 GPU 使用情况：

```bash
# 查看容器 GPU 使用
nvidia-smi

# 查看容器日志
docker-compose logs -f whisper-service
```

### 批量处理

对于批量转录，建议：
1. 使用并发请求（注意 GPU 显存限制）
2. 使用较大的模型提高准确度
3. 调整容器资源限制

## 模型缓存

首次运行时，模型会自动下载到容器内的 `/root/.cache/whisper`。使用 Docker volume 缓存模型：

```yaml
volumes:
  - whisper-cache:/root/.cache/whisper
```

## 常见问题

### 1. 容器无法访问 GPU

检查 NVIDIA Container Toolkit 是否正确安装：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. 显存不足

- 使用更小的模型（`tiny` 或 `base`）
- 限制并发请求数量
- 在 docker-compose.yml 中设置 GPU 显存限制

### 3. 音频格式不支持

确保安装了 ffmpeg（Dockerfile 中已包含）。支持的格式：
- wav, mp3, m4a, ogg, flac, mp4, mpeg, webm

### 4. 模型下载失败

首次启动需要下载模型，可能需要几分钟。检查网络连接或使用代理。

## 开发

### 本地运行（不使用 Docker）

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python main.py
```

### 运行测试

```bash
# 确保服务正在运行
python test_client.py --health
python test_client.py --file test_audio.wav
```

## 停止服务

```bash
docker-compose down

# 同时删除缓存卷
docker-compose down -v
```

## 查看日志

```bash
# 查看所有日志
docker-compose logs

# 实时查看日志
docker-compose logs -f

# 查看最近 100 行日志
docker-compose logs --tail=100
```

## 项目结构

```
whisper_service/
├── main.py                 # FastAPI 主应用
├── test_client.py          # STT 测试客户端
├── test_tts_client.py      # TTS 测试客户端
├── Dockerfile              # Docker 镜像定义
├── docker-compose.yml      # Docker Compose 配置
├── requirements.txt        # Python 依赖
├── pyproject.toml         # 项目配置
└── README.md              # 项目文档
```

## License

MIT License

## 参考

- [OpenAI Whisper](https://github.com/openai/whisper)
- [edge-tts](https://github.com/rany2/edge-tts)
- [FastAPI](https://fastapi.tiangolo.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/index.html)