import traceback
import numpy as np
import os
import sounddevice as sd
import tempfile
import wave
import webrtcvad
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI
import keyboard
import torch
import logging

from src.configurations.recording_mode import RecordingMode


class TranscriptionService:
    def __init__(self, config):
        self.config = config
        self.local_model = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger: logging.Logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def create_local_model(self):
        local_model_options = self.config.local_model_options
        device = local_model_options.device if torch.cuda.is_available() and local_model_options.device != 'cpu' else 'cpu'
        try:
            model = WhisperModel(local_model_options.model, device=device, compute_type=local_model_options.compute_type)
        except Exception as e:
            self.logger.error(f'Error initializing WhisperModel with CUDA: {e}')
            self.logger.info('Falling back to CPU.')
            model = WhisperModel(local_model_options.model, device='cpu', compute_type=local_model_options.compute_type)
        return model

    def transcribe_local(self, temp_audio_file):
        if not self.local_model:
            self.local_model = self.create_local_model()
        model_options = self.config.local_model_options
        response = self.local_model.transcribe(
            audio=temp_audio_file,
            language=model_options.language,
            initial_prompt=model_options.initial_prompt,
            condition_on_previous_text=model_options.condition_on_previous_text,
            temperature=model_options.temperature,
            vad_filter=model_options.vad_filter
        )
        return ''.join([segment.text for segment in list(response[0])])

    def transcribe_api(self, temp_audio_file):
        load_dotenv()
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        )
        api_options = self.config.api_options
        with open(temp_audio_file, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model=api_options.model,
                file=audio_file,
                language=api_options.language,
                prompt=api_options.initial_prompt,
                temperature=api_options.temperature
            )
        return response.text

    def record(self, status_queue, cancel_flag):
        sound_device = self.config.sound_device
        sample_rate = self.config.sample_rate
        frame_duration = 30
        buffer_duration = 300
        silence_duration = self.config.silence_duration

        vad = webrtcvad.Vad(3)
        buffer = []
        recording = []
        num_silent_frames = 0
        num_buffer_frames = buffer_duration // frame_duration
        num_silence_frames = silence_duration // frame_duration

        try:
            self.logger.info('Recording...')
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=sample_rate * frame_duration // 1000,
                                device=sound_device, callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0])):
                while not cancel_flag():
                    if len(buffer) < sample_rate * frame_duration // 1000:
                        continue

                    frame = buffer[:sample_rate * frame_duration // 1000]
                    buffer = buffer[sample_rate * frame_duration // 1000:]

                    if not cancel_flag():
                        if self.config.recording_mode == RecordingMode.PRESS_TO_TOGGLE.value:
                            if len(recording) > 0 and keyboard.is_pressed(self.config.activation_key):
                                break
                            else:
                                recording.extend(frame)
                        if self.config.recording_mode == RecordingMode.HOLD_TO_RECORD.value:
                            if keyboard.is_pressed(self.config.activation_key):
                                recording.extend(frame)
                            else:
                                break
                        elif self.config.recording_mode == RecordingMode.VOICE_ACTIVITY_DETECTION.value:
                            is_speech = vad.is_speech(np.array(frame).tobytes(), sample_rate)
                            if is_speech:
                                recording.extend(frame)
                                num_silent_frames = 0
                            else:
                                if len(recording) > 0:
                                    num_silent_frames += 1
                                if num_silent_frames >= num_silence_frames:
                                    break

            if cancel_flag():
                status_queue.put(('cancel', ''))
                return ''

            audio_data = np.array(recording, dtype=np.int16)
            self.logger.info(f'Recording finished. Size: {audio_data.size}')

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
                with wave.open(temp_audio_file.name, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())

            return temp_audio_file.name

        except Exception as e:
            traceback.print_exc()
            status_queue.put(('error', 'Error'))
            return ''

    def post_process_transcription(self, transcription):
        transcription = transcription.strip()
        if self.config.remove_trailing_period and transcription.endswith('.'):
            transcription = transcription[:-1]
        if self.config.add_trailing_space:
            transcription += ' '
        if self.config.remove_capitalization:
            transcription = transcription.lower()
        self.logger.info(f'Post-processed transcription: {transcription}')
        return transcription

    def transcribe(self, status_queue, cancel_flag, audio_file):
        if not audio_file:
            return ''

        status_queue.put(('transcribing', 'Transcribing...'))
        self.logger.info('Transcribing audio file...')

        if self.config.use_api:
            transcription = self.transcribe_api(audio_file)
        else:
            transcription = self.transcribe_local(audio_file)

        self.logger.info(f'Transcription: {transcription}')
        return self.post_process_transcription(transcription)

    def record_and_transcribe(self, status_queue, cancel_flag):
        audio_file = self.record(status_queue, cancel_flag)
        if cancel_flag():
            return ''
        result = self.transcribe(status_queue, cancel_flag, audio_file)
        return result
