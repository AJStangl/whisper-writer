import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import keyboard
from audioplayer import AudioPlayer
from pynput.keyboard import Controller

from status_window import StatusWindow
from transcription import create_local_model, record_and_transcribe


class WhisperWriter:
    def __init__(self) -> None:
        self.config: Dict[str, Any] = self.load_config_with_defaults()
        self.status_queue: queue.Queue = queue.Queue()
        self.local_model: Optional[Any] = None
        self.pyinput_keyboard: Controller = Controller()
        self.logger: logging.Logger = self.setup_logger()

        if not self.config['use_api']:
            self.logger.info('Creating local model...')
            self.local_model = create_local_model(self.config)
            self.logger.info('Local model created.')

        self.setup()

    class ResultThread(threading.Thread):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.result: Optional[Any] = None
            self.stop_transcription: bool = False

        def run(self) -> None:
            self.result = self._target(*self._args, cancel_flag=lambda: self.stop_transcription, **self._kwargs)

        def stop(self) -> None:
            self.stop_transcription = True

    def load_config_with_defaults(self) -> Dict[str, Any]:
        default_config: Dict[str, Any] = {
            'use_api': False,
            'api_options': {
                'model': 'whisper-1',
                'language': None,
                'temperature': 0.0,
                'initial_prompt': None
            },
            'local_model_options': {
                'model': 'base',
                'device': 'auto',
                'compute_type': 'auto',
                'language': None,
                'temperature': 0.0,
                'initial_prompt': None,
                'condition_on_previous_text': True,
                'vad_filter': False,
            },
            'activation_key': 'ctrl+shift+space',
            'recording_mode': 'voice_activity_detection',
            # 'voice_activity_detection', 'press_to_toggle', continuous, or 'hold_to_record'
            'sound_device': None,
            'sample_rate': 16000,
            'silence_duration': 900,
            'writing_key_press_delay': 0.008,
            'noise_on_completion': False,
            'remove_trailing_period': True,
            'add_trailing_space': False,
            'remove_capitalization': False,
            'print_to_terminal': True,
            'hide_status_window': False
        }

        config_path: str = os.path.join('src', 'config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as config_file:
                user_config: Dict[str, Any] = json.load(config_file)
                for key, value in user_config.items():
                    if key in default_config and value is not None:
                        default_config[key] = value

        return default_config

    def clear_status_queue(self) -> None:
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break

    def on_shortcut(self) -> None:
        self.clear_status_queue()

        self.status_queue.put(('recording', 'Recording...'))
        recording_thread: WhisperWriter.ResultThread = self.ResultThread(
            target=record_and_transcribe,
            args=(self.status_queue,),
            kwargs={'config': self.config,
                    'local_model': self.local_model if self.local_model and not self.config['use_api'] else None}
        )

        if not self.config['hide_status_window']:
            status_window: StatusWindow = StatusWindow(self.status_queue)
            status_window.recording_thread = recording_thread
            status_window.start()

        recording_thread.start()
        recording_thread.join()

        if not self.config['hide_status_window']:
            if status_window.is_alive():
                self.status_queue.put(('cancel', ''))

        transcribed_text: Optional[str] = recording_thread.result

        if transcribed_text:
            self.typewrite(transcribed_text, interval=self.config['writing_key_press_delay'])

        if self.config['noise_on_completion']:
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

    def format_keystrokes(self, key_string: str) -> str:
        return '+'.join(word.capitalize() for word in key_string.split('+'))

    def typewrite(self, text: str, interval: float) -> None:
        for letter in text:
            self.pyinput_keyboard.press(letter)
            self.pyinput_keyboard.release(letter)
            time.sleep(interval)

    def setup(self) -> None:
        model_method: str = 'OpenAI\'s API' if self.config['use_api'] else 'a local model'
        self.logger.info(
            f'Script activated. Whisper is set to run using {model_method}. To change this, modify the "use_api" value in the src\\config.json file.')

        self.logger.info(
            f'WhisperWriter is set to record using {self.config["recording_mode"]}. To change this, modify the "recording_mode" value in the src\\config.json file.')
        self.logger.info(f'The activation key combo is set to {self.format_keystrokes(self.config["activation_key"])}.')

        if self.config['recording_mode'] == 'voice_activity_detection':
            self.logger.info(' When it is pressed, recording will start, and will stop when you stop speaking.')
        elif self.config['recording_mode'] == 'press_to_toggle':
            self.logger.info(
                ' When it is pressed, recording will start, and will stop when you press the key combo again.')
        elif self.config['recording_mode'] == 'hold_to_record':
            self.logger.info(' When it is pressed, recording will start, and will stop when you release the key combo.')

        keyboard.add_hotkey(self.config['activation_key'], self.on_shortcut)
        try:
            keyboard.wait()  # Keep the script running to listen for the shortcut
        except KeyboardInterrupt:
            self.logger.info('\nExiting the script...')
            os.system('exit')

    @staticmethod
    def setup_logger() -> logging.Logger:
        logger: logging.Logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler: logging.StreamHandler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


if __name__ == "__main__":
    WhisperWriter()
