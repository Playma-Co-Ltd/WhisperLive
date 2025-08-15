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
        self.language = "auto"
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
        print("轉譯參數資料",self.no_speech_thresh,self.same_output_threshold)
        # ensure mono float32 in [-1, 1]
        x = np.asarray(audio_chunk, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        
        # 檢測是否為靜音 - 計算音訊能量
        energy = np.sqrt(np.mean(x**2))
        silence_threshold = 0.03  # 可調整的靜音閾值
        
        # 如果音訊能量低於閾值，判定為靜音，不進行轉譯
        if energy < silence_threshold:
            print("判斷為靜音")
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
            # 呼叫 OpenAI Whisper API with context
            api_params = {
                "model": self.model,
                "file": buffer,
                "temperature": 0,
                "prompt": "",
                "response_format": "verbose_json", # Get timestamps
            }
            
            # Add language if specified
            if self.language and self.language != "auto":
                api_params["language"] = self.language
            
            response = self.client.audio.transcriptions.create(**api_params)
            # print("單次轉譯資料",response)
            print("轉譯完成")
            
            # Return response object for processing
            return response
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
        if result is None:
            return
        language = result.language
        result = result.segments
        
        segments = []
        if len(result):
            self.t_start = None
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments,language)


    def send_transcription_to_client(self, segments,language):
        """
        改寫 base.py 的 method，將 whisper 辨識出的語言透過 language 送給 client
        """
        for s in segments:
            s["source_language"] = language
        
        # print("準備要送給 client 的資料",segments)

        try:
            self.websocket.send(
                json.dumps({
                    "uid": self.client_uid,
                    "segments": segments
                })
            )
            print("發送完成")
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")