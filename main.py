import os
import tempfile
from pathlib import Path
from typing import Optional, Literal
from io import BytesIO

import torch
import whisper
import edge_tts
import opencc
import ChatTTS
import numpy as np
import soundfile as sf
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
chat_tts = None

HALLUCINATION_SET = {
    "thank you",
    "thanks",
    "thank you.",
    "okay",
    "ok",
    "bye",
    "goodbye",
    "嗯",
    "啊",
    "哦",
    "好的",
    "谢谢",
    "再见",
}

def is_hallucination(text: str) -> bool:
    # Check if any part of the text (split by Chinese period) is in the hallucination set
    parts = text.lower().strip(".").strip(",").strip("?").split("。")
    for part in parts:
        if part.strip() in HALLUCINATION_SET:
            return True
    return False


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


def load_chat_tts():
    """加载 Chat TTS 模型"""
    global chat_tts
    if chat_tts is None:
        print(f"Loading Chat TTS model on {DEVICE}...")
        chat_tts = ChatTTS.Chat()
        chat_tts.load_models(device=DEVICE)
        print(f"Chat TTS model loaded successfully!")
    return chat_tts


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()
    load_chat_tts()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        model_size=MODEL_SIZE,
        device=DEVICE,
        cuda_available=torch.cuda.is_available()
    )


@app.post("/stt/transcribe", response_model=TranscribeResponse)
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
            "task": task
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

        if is_hallucination(text):
            text = ""

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


@app.get("/tts/voices", response_model=VoicesResponse)
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


@app.post("/tts/synthesize")
async def text_to_speech(
    text: str = Form(..., description="要转换的文本"),
    voice: int = Form(0, description="音色ID (0: 女声, 1: 男声)"),
    speed: float = Form(1.0, description="语速 (0.5-2.0, 默认1.0)"),
    temperature: float = Form(0.3, description="随机性 (0.1-1.0, 默认0.3)"),
    top_p: float = Form(0.7, description="top_p采样 (0.1-1.0, 默认0.7)"),
    top_k: int = Form(20, description="top_k采样 (1-50, 默认20)"),
):
    """
    文字转语音 (Text-to-Speech) 使用 Chat TTS

    返回 WAV 格式的音频文件

    参数说明:
    - text: 要转换的文本内容
    - voice: 音色ID
      - 0: 女声 (默认)
      - 1: 男声
    - speed: 语速控制，范围 0.5-2.0，默认 1.0
    - temperature: 随机性控制，范围 0.1-1.0，默认 0.3，值越高越随机
    - top_p: 核采样参数，范围 0.1-1.0，默认 0.7
    - top_k: top_k采样参数，范围 1-50，默认 20
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 characters)")

    # 验证参数范围
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")
    if not 0.1 <= temperature <= 1.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0.1 and 1.0")
    if not 0.1 <= top_p <= 1.0:
        raise HTTPException(status_code=400, detail="Top_p must be between 0.1 and 1.0")
    if not 1 <= top_k <= 50:
        raise HTTPException(status_code=400, detail="Top_k must be between 1 and 50")
    if voice not in [0, 1]:
        raise HTTPException(status_code=400, detail="Voice must be 0 (female) or 1 (male)")

    try:
        # 加载 Chat TTS 模型
        chat_tts_model = load_chat_tts()

        # 设置随机种子以获得一致的结果
        seed = 42  # 固定种子，可以根据需要改为随机数

        # 生成音频
        wavs = chat_tts_model.infer(
            [text],
            skip_refine_text=True,
            params_infer_code={
                'spk_emb': chat_tts_model.spk_emb[voice] if hasattr(chat_tts_model, 'spk_emb') else None,
                'temperature': temperature,
                'top_P': top_p,
                'top_K': top_k,
                'manual_seed': seed,
            },
            params_refine_text={
                'prompt': '[oral_2][laugh_0][break_6]',
            },
        )

        # 获取生成的音频
        audio_array = wavs[0]

        # 应用速度调整（通过重采样）
        if speed != 1.0:
            import librosa
            audio_array = librosa.effects.time_stretch(audio_array, rate=speed)

        # 将音频转换为 WAV 格式
        # 创建临时文件来保存 WAV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            sf.write(temp_wav.name, audio_array, 24000)  # Chat TTS 使用 24kHz 采样率
            temp_wav_path = temp_wav.name

        # 读取 WAV 文件
        with open(temp_wav_path, 'rb') as f:
            audio_data = f.read()

        # 清理临时文件
        os.remove(temp_wav_path)

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        # 返回音频流
        return StreamingResponse(
            BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.wav",
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