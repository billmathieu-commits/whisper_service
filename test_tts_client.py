#!/usr/bin/env python3
"""
TTS 服务测试脚本
用于测试 Edge-TTS 文字转语音服务
"""

import requests
import sys
import argparse
from pathlib import Path


def list_voices(base_url: str = "http://localhost:8000", language: str = None):
    """列出可用的声音"""
    print("=" * 50)
    print("获取可用声音列表...")
    print("=" * 50)

    try:
        params = {}
        if language:
            params["language"] = language
            print(f"筛选语言: {language}")

        response = requests.get(f"{base_url}/tts/voices", params=params)
        response.raise_for_status()
        data = response.json()

        print(f"\n找到 {data['count']} 个声音:\n")

        # 按语言分组
        voices_by_lang = {}
        for voice in data['voices']:
            lang = voice['language']
            if lang not in voices_by_lang:
                voices_by_lang[lang] = []
            voices_by_lang[lang].append(voice)

        # 显示声音
        for lang, voices in sorted(voices_by_lang.items()):
            print(f"\n{lang}:")
            for voice in voices[:5]:  # 每种语言只显示前5个
                print(f"  - {voice['name']} ({voice['gender']})")
                if voice.get('description'):
                    print(f"    {voice['description']}")
            if len(voices) > 5:
                print(f"  ... 还有 {len(voices) - 5} 个声音")

        print()
        return True

    except Exception as e:
        print(f"✗ 获取声音列表失败: {e}")
        print()
        return False


def synthesize_speech(
    text: str,
    base_url: str = "http://localhost:8000",
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    volume: str = "+0%",
    pitch: str = "+0Hz",
    output_file: str = "output.mp3"
):
    """合成语音"""
    print("=" * 50)
    print("文字转语音...")
    print("=" * 50)
    print(f"文本: {text}")
    print(f"声音: {voice}")
    print(f"语速: {rate}")
    print(f"音量: {volume}")
    print(f"音调: {pitch}")
    print(f"输出文件: {output_file}")
    print()

    try:
        data = {
            "text": text,
            "voice": voice,
            "rate": rate,
            "volume": volume,
            "pitch": pitch
        }

        print("正在合成语音...")
        response = requests.post(f"{base_url}/tts/synthesize", data=data)

        if response.status_code == 200:
            # 保存音频文件
            with open(output_file, "wb") as f:
                f.write(response.content)

            file_size = len(response.content)
            print(f"✓ 合成成功!")
            print(f"✓ 文件大小: {file_size / 1024:.2f} KB")
            print(f"✓ 已保存到: {output_file}")
            print()
            return True
        else:
            print(f"✗ 合成失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            print()
            return False

    except Exception as e:
        print(f"✗ 请求失败: {e}")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description="TTS 服务测试客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="服务地址")
    parser.add_argument("--list-voices", action="store_true", help="列出所有可用声音")
    parser.add_argument("--text", help="要转换的文本")
    parser.add_argument("--voice", default="zh-CN-XiaoxiaoNeural", help="TTS 声音名称")
    parser.add_argument("--rate", default="+0%", help="语速调整 (如: +10%%, -20%%)")
    parser.add_argument("--volume", default="+0%", help="音量调整 (如: +10%%, -20%%)")
    parser.add_argument("--pitch", default="+0Hz", help="音调调整 (如: +50Hz, -50Hz)")
    parser.add_argument("--output", default="output.mp3", help="输出音频文件名")
    parser.add_argument("--language", help="按语言筛选声音列表 (如: zh-CN, en-US)")

    args = parser.parse_args()

    # 列出声音
    if args.list_voices:
        if not list_voices(args.url, args.language):
            sys.exit(1)
        sys.exit(0)

    # 如果没有提供文本，显示帮助
    if not args.text:
        parser.print_help()
        print("\n示例用法:")
        print("  # 列出所有声音")
        print("  python test_tts_client.py --list-voices")
        print("  python test_tts_client.py --list-voices --language zh-CN")
        print()
        print("  # 合成语音")
        print('  python test_tts_client.py --text "你好，世界！"')
        print('  python test_tts_client.py --text "Hello, world!" --voice en-US-JennyNeural')
        print('  python test_tts_client.py --text "你好" --rate=+20% --volume=+10%')
        sys.exit(1)

    # 合成语音
    if not synthesize_speech(
        args.text,
        args.url,
        args.voice,
        args.rate,
        args.volume,
        args.pitch,
        args.output
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()