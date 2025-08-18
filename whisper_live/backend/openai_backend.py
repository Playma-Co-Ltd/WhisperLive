# whisper_live/backend/openai_backend.py
import json
import logging
import threading
import numpy as np
import io, wave
import time
from openai import OpenAI

from whisper_live.backend.base import ServeClientBase

from dotenv import load_dotenv

load_dotenv()

class ServeClientOpenAI(ServeClientBase):
    def __init__(
        self,
        websocket,
        client_uid,
        model="whisper-1",
        language="auto",
        task="transcribe",
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
        initial_prompt=None,
    ):
        super().__init__(
            client_uid=client_uid,
            websocket=websocket,
            send_last_n_segments=send_last_n_segments,
            no_speech_thresh=no_speech_thresh,
            clip_audio=clip_audio,
            same_output_threshold=same_output_threshold,
            translation_queue=translation_queue,
        )
        self.client = OpenAI()
        self.model = "whisper-1"
        self.language = language
        self.task = task
        self.initial_prompt = initial_prompt

        
        logging.info(f"OpenAI ASR backend initialized with model: {self.model}, language: {self.language}")
        
        # Start transcription thread
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        
        # Send SERVER_READY message to client
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "openai"
                }
            )
        )
    
    def transcribe_audio(self, audio_chunk: np.ndarray):
        # Process audio directly without buffering to match faster_whisper behavior
        # This ensures consistent timing and better quality
        
        # ensure mono float32 in [-1, 1]
        x = np.asarray(audio_chunk, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        
        # Silence dectection
        energy = np.sqrt(np.mean(x**2))
        silence_threshold = 0.03 
        if energy < silence_threshold:
            return None

        # convert to 16-bit PCM
        pcm16 = (x * 32767.0).astype(np.int16)

        # write a proper PCM16 WAV header + data
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(self.RATE))  # e.g., 16000
            wf.writeframes(pcm16.tobytes())

        buffer.seek(0)
        buffer.name = "audio.wav"  # important for content-type sniffing

        try:
            # Don’t pass language parameters when the client selects auto-detection.
            # Before each transcription, language is set to None to enable auto-detection.
            # If the user passes a language code, then all transcriptions will use that language code.
            api_params = {
                "model": self.model,
                "file": buffer,
                "temperature": 0,
                "prompt": self.initial_prompt,
                "language":self.language,
                "response_format": "verbose_json", # Get timestamps
            }

            if self.language == "auto":
                api_params.pop("language", None)
            
            result = self.client.audio.transcriptions.create(**api_params)

            # Update the dectected language code from whisper
            if result.language:
                self.dectected_language = result.language
            
            return result

        except Exception as e:
            logging.error(f"OpenAI API transcription error: {e}")
            return None

    def handle_transcription_output(self, result, duration: float):
        
        """
            Handle the transcription output, updating the transcript and sending data to the client.

            Args:
                result (str): The result from whisper inference i.e. the list of segments.
                duration (float): Duration of the transcribed audio chunk.
        """

        # If the ASR process for a chunk of audio encounters a network error, do not perform any action.
        # Ignore this transcription.
        if result is None:
            return
        result = result.segments
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)


    def send_transcription_to_client(self, segments):
        """
        Modify the method in base.py to send the language detected by Whisper to the client via the language field.
        """
        for s in segments:
            s["source_language"] = self.dectected_language 
        
        logging.info(f"For client {segments}")

        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments
                })
            )
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")