import os
import tempfile
from pathlib import Path
from typing import Optional, Literal
from io import BytesIO

import torch
import whisper
import edge_tts
import opencc
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# 配置
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Speech Service",
    description="Speech-to-Text (Whisper) and Text-to-Speech (Edge TTS) service with GPU acceleration",
    version="1.0.0"
)

# 全局模型变量
model = None


class TranscribeResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model_size: str
    device: str
    cuda_available: bool


class VoiceInfo(BaseModel):
    name: str
    gender: str
    language: str
    description: Optional[str] = None


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    count: int


def load_model():
    """加载 Whisper 模型"""
    global model
    if model is None:
        print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
        model = whisper.load_model(MODEL_SIZE, device=DEVICE)
        print(f"Model loaded successfully!")
    return model


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        model_size=MODEL_SIZE,
        device=DEVICE,
        cuda_available=torch.cuda.is_available()
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(..., description="音频文件"),
    language: Optional[str] = Form(None, description="语言代码 (如: zh, en, ja)，自动检测则为空"),
    task: Literal["transcribe", "translate"] = Form("transcribe", description="任务类型：transcribe(转录) 或 translate(翻译成英文)"),
):
    """
    转录音频文件

    支持的音频格式：wav, mp3, m4a, ogg, flac, 等 ffmpeg 支持的格式
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # 验证文件扩展名
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".mp4", ".mpeg", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {', '.join(allowed_extensions)}"
        )

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # 加载模型
        whisper_model = load_model()

        # 转录音频
        options = {
            "task": task,
        }

        # 如果指定了语言，添加到选项中
        if language:
            options["language"] = language

        result = whisper_model.transcribe(temp_file_path, **options)

        # 获取音频时长
        audio = whisper.load_audio(temp_file_path)
        duration = len(audio) / whisper.audio.SAMPLE_RATE

        # 转换为简体中文
        text = result["text"].strip()
        detected_language = result.get("language")

        # 如果是中文，转换为简体
        if detected_language and detected_language.startswith("zh"):
            try:
                cc = opencc.OpenCC('t2s')  # 繁体转简体
                text = cc.convert(text)
            except Exception:
                # 如果转换失败，使用原文
                pass

        return TranscribeResponse(
            text=text,
            language=detected_language,
            duration=duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/voices", response_model=VoicesResponse)
async def list_voices(
    language: Optional[str] = Query(None, description="按语言筛选 (如: zh-CN, en-US, ja-JP)")
):
    """
    列出所有可用的 TTS 声音

    参数:
    - language: 可选，按语言筛选 (如: zh-CN, en-US, ja-JP)
    """
    try:
        voices = await edge_tts.list_voices()

        # 筛选声音
        if language:
            voices = [v for v in voices if v.get("Locale", "").startswith(language)]

        # 格式化响应
        voice_list = []
        for voice in voices:
            voice_list.append(VoiceInfo(
                name=voice.get("Name", ""),
                gender=voice.get("Gender", ""),
                language=voice.get("Locale", ""),
                description=voice.get("Description", "")
            ))

        return VoicesResponse(voices=voice_list, count=len(voice_list))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")


@app.post("/synthesize")
async def text_to_speech(
    text: str = Form(..., description="要转换的文本"),
    voice: str = Form("zh-CN-XiaoxiaoNeural", description="TTS 声音名称"),
    rate: str = Form("+0%", description="语速调整 (如: +10%, -20%)"),
    volume: str = Form("+0%", description="音量调整 (如: +10%, -20%)"),
    pitch: str = Form("+0Hz", description="音调调整 (如: +50Hz, -50Hz)"),
):
    """
    文字转语音 (Text-to-Speech)

    返回 MP3 格式的音频文件

    常用声音:
    - 中文女声: zh-CN-XiaoxiaoNeural
    - 中文男声: zh-CN-YunyangNeural
    - 英文女声: en-US-JennyNeural
    - 英文男声: en-US-GuyNeural
    - 日文女声: ja-JP-NanamiNeural

    使用 /tts/voices 查看所有可用声音
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    try:
        # 创建 TTS communicate 对象
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch
        )

        # 生成音频数据
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # 返回音频流
        return StreamingResponse(
            BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.mp3",
                "Content-Length": str(len(audio_data))
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Speech Service (STT + TTS)",
        "endpoints": {
            "health": "/health",
            "stt": "/transcribe (POST)",
            "tts": "/tts/synthesize (POST)",
            "voices": "/tts/voices",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)