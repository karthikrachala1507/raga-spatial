"""
Raga Spatial - core/audio_io.py
One way to read any uploaded song.

libsndfile reads WAV, FLAC, OGG and MP3 but not M4A or AAC, and the audioread
fallback inside librosa finds no decoder on this machine, so those uploads used
to fail at step 1. They are now decoded with the ffmpeg from static-ffmpeg.
"""
import os
import shutil
import subprocess
import tempfile

import librosa
import soundfile as sf


def ffmpeg_exe():
    # The bundled static build comes first: the ffmpeg inside the conda environment
    # (first on PATH once it is activated) does not start on this machine.
    import static_ffmpeg
    bundled = os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin", "win32", "ffmpeg.exe")
    if os.path.exists(bundled):
        return bundled            # asking static_ffmpeg itself can block on its lock file
    found = shutil.which("ffmpeg")
    if found:
        return found
    from static_ffmpeg import run
    return run.get_or_fetch_platform_executables_else_raise()[0]


def load(path, sr, mono=True):
    """Same result as librosa.load, for every format the app accepts."""
    try:
        sf.info(path)
        return librosa.load(path, sr=sr, mono=mono)
    except sf.LibsndfileError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        run = subprocess.run([ffmpeg_exe(), "-y", "-v", "error", "-i", path, "-vn",
                              "-c:a", "pcm_f32le", wav], capture_output=True, text=True, errors="replace",
                             stdin=subprocess.DEVNULL, timeout=600)
        if run.returncode != 0 or not os.path.exists(wav):
            raise ValueError("Could not read this audio file: " + run.stderr.strip()[-300:])
        return librosa.load(wav, sr=sr, mono=mono)
