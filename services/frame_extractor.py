import cv2
import os
import subprocess
from config import FRAME_DIR, AUDIO_DIR, FRAME_EXTRACT_RATE


def extract_frames(video_path: str, session_id: str) -> str:
    """
    Extracts frames from the video at a regular interval.
    Returns the path to the directory containing extracted frames.
    """

    session_frame_path = FRAME_DIR / session_id
    os.makedirs(session_frame_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if FPS is 0
    interval = max(int(fps * FRAME_EXTRACT_RATE), 1)

    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame_name = f"frame_{frame_id}.jpg"
            cv2.imwrite(str(session_frame_path / frame_name), frame)
            frame_id += 1

        count += 1

    cap.release()
    return str(session_frame_path)


def get_audio_from_video(video_path: str, session_id: str) -> str:
    """
    Extracts audio from video file using FFmpeg.
    Returns the path to the extracted audio file.
    """

    session_audio_path = AUDIO_DIR / session_id
    os.makedirs(session_audio_path, exist_ok=True)

    output_audio = str(session_audio_path / "audio.mp3")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "libmp3lame",
        "-y",
        output_audio
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return output_audio

    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:")
        print(e.stderr.decode())  # 👈 VERY IMPORTANT for debugging
        return ""