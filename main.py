import os
import tempfile
from pathlib import Path
from typing import Optional, Literal

import torch
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 配置
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Whisper STT Service",
    description="Speech-to-Text service using OpenAI Whisper with GPU acceleration",
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

        return TranscribeResponse(
            text=result["text"].strip(),
            language=result.get("language"),
            duration=duration
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Whisper STT Service",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)