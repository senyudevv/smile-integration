import pyaudio
import audioop
import time
import json
import re
from vosk import Model, KaldiRecognizer
import pyttsx3

from src.config import VOSK_MODEL_PATH, VOICE_RATE, VOICE_VOLUME, DEFAULT_VOICE_INDEX, MIC_INDEX, MIC_SAMPLE_RATE
model = Model(str(VOSK_MODEL_PATH))


def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', VOICE_RATE)
    engine.setProperty('volume', VOICE_VOLUME)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[DEFAULT_VOICE_INDEX].id)
    engine.say(text)
    engine.runAndWait()


def find_working_mic(trials_rates=(16000, 48000), trials_channels=(1, 2), timeout=1.0):
    """
    Cherche automatiquement un périphérique audio utilisable.
    Retourne (device_index, rate, channels) ou (None, None, None).
    """
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] <= 0:
            continue
        for rate in trials_rates:
            for ch in trials_channels:
                try:
                    stream = p.open(format=pyaudio.paInt16,
                                    channels=ch,
                                    rate=rate,
                                    input=True,
                                    input_device_index=i,
                                    frames_per_buffer=1024)
                    try:
                        data = stream.read(1024, exception_on_overflow=False)
                        if data and len(data) > 0:
                            stream.stop_stream()
                            stream.close()
                            p.terminate()
                            return i, rate, ch
                    except Exception:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
    p.terminate()
    return None, None, None


def _to_mono_and_resample(raw_bytes, width, in_channels, in_rate, out_rate=16000):
    """Convertit des bytes PCM bruts en mono 16kHz pour Vosk."""
    if in_channels == 2:
        mono = audioop.tomono(raw_bytes, width, 0.5, 0.5)
    elif in_channels == 1:
        mono = raw_bytes
    else:
        # >2 canaux : downmix simple
        try:
            mono = audioop.tomono(raw_bytes, width, 1.0 / in_channels, 1.0 / in_channels)
        except Exception:
            mono = raw_bytes
    if in_rate != out_rate:
        try:
            converted, _ = audioop.ratecv(mono, width, 1, in_rate, out_rate, None)
            return converted
        except Exception:
            return mono
    return mono


def transcribe_audio(duration=20, stop_on_silence=True, silence_limit=1.5, silence_hangover=2.2):
    """
    Enregistrement audio robuste.
    Retourne la chaîne transcrite ou "" en cas d'échec.
    """
    dev_idx, dev_rate, dev_ch = find_working_mic()
    if dev_idx is None:
        print("❌ [MIC] Aucun microphone accessible trouvé.")
        return ""

    RATE = dev_rate
    CHANNELS = dev_ch
    import pyaudio as _pa
    _p = _pa.PyAudio()
    print(f"🎙️ [MIC] Micro sélectionné : [{dev_idx}] {_p.get_device_info_by_index(dev_idx)['name']} @ {RATE}Hz")
    _p.terminate()
    CHUNK = 2048
    FORMAT = pyaudio.paInt16

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=dev_idx,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"⚠️ [MIC] Impossible d'ouvrir le flux (index={dev_idx}) : {e}")
        p.terminate()
        return ""

    frames = []
    last_voice_time = None
    speech_detected = False
    start_time = time.time()

    try:
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                time.sleep(0.01)
                continue

            if not data:
                break

            frames.append(data)
            try:
                rms = audioop.rms(data, 2)
            except Exception:
                rms = 0

            print(f"[MIC] RMS={rms}", end="\r")

            if rms > 80:
                speech_detected = True
                last_voice_time = time.time()
            else:
                if speech_detected:
                    if last_voice_time and (time.time() - last_voice_time) > silence_hangover:
                        print("\n🔇 [MIC] Silence détecté, fermeture du micro.")
                        break
                else:
                    if time.time() - start_time > duration:
                        print("\n⏱️ [MIC] Durée maximale atteinte, fermeture du micro.")
                        break

            if time.time() - start_time > duration:
                print("\n⏱️ [MIC] Durée maximale atteinte, fermeture du micro.")
                break

    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

    raw = b"".join(frames)
    try:
        mono16 = _to_mono_and_resample(raw, 2, CHANNELS, RATE, out_rate=16000)
    except Exception as e:
        print(f"⚠️ [MIC] Erreur de conversion audio : {e}")
        mono16 = raw

    try:
        rec = KaldiRecognizer(model, 16000)
        offset = 0
        step = 4000
        text = ""
        while offset < len(mono16):
            chunk = mono16[offset:offset + step]
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                text += " " + res.get("text", "")
            offset += step
        text += " " + json.loads(rec.FinalResult()).get("text", "")
        text = text.strip()
    except Exception as e:
        print(f"⚠️ [STT] Erreur Vosk : {e}")
        text = ""

    return text


def extract_name_from_text(text: str) -> str:
    text = text.lower().strip()
    blacklist = {"bonjour", "salut", "je", "m'appelle", "appelle", "suis", "moi", "c'est", "le", "la", "un", "une", "oui", "non"}

    text = re.sub(r"[^a-zàâäéèêëîïôùûüœç\s]", "", text)

    m = re.search(r"(?:je m'?appelle|mappelle|je suis|m'?appelle|c'?est)\s+([a-zàâäéèêëîïôùûüœç]+)", text)
    if m:
        name = m.group(1).capitalize()
    else:
        words = [w for w in text.split() if w not in blacklist]
        if not words:
            return f"Utilisateur_{int(time.time())}"
        candidates = [w for w in words if w not in ["merci", "voilà", "bien"]]
        name = candidates[0].capitalize() if candidates else f"Utilisateur_{int(time.time())}"

    if len(name) < 2 or name.lower() in ["bien", "merci", "oui", "non", "ok"]:
        name = f"Utilisateur_{int(time.time())}"

    return name
