#!/usr/bin/env python3
"""
示例客户端测试脚本
用于测试 Whisper STT 服务
"""

import requests
import sys
from pathlib import Path


def test_health(base_url: str = "http://localhost:8000"):
    """测试健康检查接口"""
    print("=" * 50)
    print("测试健康检查接口...")
    print("=" * 50)

    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ 服务状态: {data['status']}")
        print(f"✓ 模型大小: {data['model_size']}")
        print(f"✓ 设备: {data['device']}")
        print(f"✓ CUDA 可用: {data['cuda_available']}")
        print()
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        print()
        return False


def test_transcribe(audio_file: str, base_url: str = "http://localhost:8000", language: str = None):
    """测试转录接口"""
    print("=" * 50)
    print("测试转录接口...")
    print("=" * 50)

    file_path = Path(audio_file)
    if not file_path.exists():
        print(f"✗ 文件不存在: {audio_file}")
        return False

    print(f"音频文件: {file_path.name}")
    print(f"文件大小: {file_path.stat().st_size / 1024:.2f} KB")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "audio/mpeg")}
            data = {}

            if language:
                data["language"] = language
                print(f"语言: {language}")

            print("正在上传和转录...")
            response = requests.post(f"{base_url}/transcribe", files=files, data=data)
            response.raise_for_status()

        result = response.json()
        print(f"\n✓ 转录成功!")
        print(f"识别文本: {result['text']}")
        if result.get('language'):
            print(f"检测语言: {result['language']}")
        if result.get('duration'):
            print(f"音频时长: {result['duration']:.2f} 秒")
        print()
        return True

    except Exception as e:
        print(f"✗ 转录失败: {e}")
        print()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Whisper STT 服务测试客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--health", action="store_true", help="仅测试健康检查")
    parser.add_argument("--file", help="要转录的音频文件路径")
    parser.add_argument("--language", help="语言代码 (如: zh, en, ja)")

    args = parser.parse_args()

    # 测试健康检查
    if not test_health(args.url):
        sys.exit(1)

    # 如果只是测试健康检查，则退出
    if args.health:
        sys.exit(0)

    # 如果没有提供文件，显示帮助
    if not args.file:
        parser.print_help()
        print("\n示例用法:")
        print("  python test_client.py --health")
        print("  python test_client.py --file audio.wav")
        print("  python test_client.py --file audio.mp3 --language zh")
        sys.exit(1)

    # 测试转录
    if not test_transcribe(args.file, args.url, args.language):
        sys.exit(1)


if __name__ == "__main__":
    main()