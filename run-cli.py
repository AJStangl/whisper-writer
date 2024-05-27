import os
import queue
import threading
import time
import logging
import keyboard
import click
from typing import Any, Optional
from audioplayer import AudioPlayer
from pynput.keyboard import Controller

from src.configurations.recording_mode import RecordingMode
from src.writer.transcription_module import TranscriptionService
from src.writer.status_window_module import StatusWindow
from src.configurations.configuration import Configuration, APIOptions, LocalModelOptions

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'


class WhisperWriterCli:
    def __init__(self, config: Configuration) -> None:
        self.config: Configuration = config
        self.status_queue: queue.Queue = queue.Queue()
        self.local_model: Optional[Any] = None
        self.pyinput_keyboard: Controller = Controller()
        self.logger: logging.Logger = self.setup_logger()
        self.transcription_service: TranscriptionService = TranscriptionService(self.config)

        if not self.config.use_api:
            self.logger.info('Creating local model...')
            self.local_model = self.transcription_service.create_local_model()
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

    def clear_status_queue(self) -> None:
        while not self.status_queue.empty():
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                break

    def on_shortcut(self) -> None:
        self.clear_status_queue()

        self.status_queue.put(('recording', 'Recording...'))
        recording_thread: WhisperWriterCli.ResultThread = self.ResultThread(
            target=self.transcription_service.record_and_transcribe,
            args=(self.status_queue,),
            kwargs={'config': self.config,
                    'local_model': self.local_model if self.local_model and not self.config.use_api else None}
        )

        if not self.config.hide_status_window:
            status_window: StatusWindow = StatusWindow(self.status_queue)
            status_window.recording_thread = recording_thread
            status_window.start()

        recording_thread.start()
        recording_thread.join()

        if not self.config.hide_status_window:
            if status_window.is_alive():
                self.status_queue.put(('cancel', ''))

        transcribed_text: Optional[str] = recording_thread.result

        if transcribed_text:
            self.typewrite(transcribed_text, interval=self.config.writing_key_press_delay)

        if self.config.noise_on_completion:
            AudioPlayer(os.path.join('assets', 'beep.wav')).play(block=True)

    def format_keystrokes(self, key_string: str) -> str:
        return '+'.join(word.capitalize() for word in key_string.split('+'))

    def typewrite(self, text: str, interval: float) -> None:
        for letter in text:
            self.pyinput_keyboard.press(letter)
            self.pyinput_keyboard.release(letter)
            time.sleep(interval)

    def setup(self) -> None:
        model_method: str = 'OpenAI\'s API' if self.config.use_api else 'a local model'
        self.logger.info(
            f'Script activated. Whisper is set to run using {model_method}. To change this, modify the "use_api" value in the src\\config.json file.')

        self.logger.info(
            f'WhisperWriter is set to record using {self.config.recording_mode.value}. To change this, modify the "recording_mode" value in the src\\config.json file.')
        self.logger.info(f'The activation key combo is set to {self.format_keystrokes(self.config.activation_key)}.')

        if self.config.recording_mode == RecordingMode.VOICE_ACTIVITY_DETECTION:
            self.logger.info(' When it is pressed, recording will start, and will stop when you stop speaking.')
        elif self.config.recording_mode == RecordingMode.PRESS_TO_TOGGLE:
            self.logger.info(
                ' When it is pressed, recording will start, and will stop when you press the key combo again.')
        elif self.config.recording_mode == RecordingMode.HOLD_TO_RECORD:
            self.logger.info(' When it is pressed, recording will start, and will stop when you release the key combo.')

        try:
            keyboard.add_hotkey(self.config.activation_key, self.on_shortcut)
            self.logger.info(f'Hotkey {self.config.activation_key} added successfully.')
        except Exception as e:
            self.logger.error(f'Failed to add hotkey {self.config.activation_key}: {e}')

        try:
            keyboard.wait()  # Keep the script running to listen for the shortcut
        except KeyboardInterrupt:
            self.logger.info('\nExiting the script...')
            os.system('exit')

    def setup_logger(self) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler: logging.StreamHandler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


@click.command()
@click.option('--use-api', is_flag=True, default=False, help='Use API instead of local model')
@click.option('--api-model', default='whisper-1', help='API model name')
@click.option('--api-language', default=None, help='API language option')
@click.option('--api-temperature', default=0.0, type=float, help='API temperature setting')
@click.option('--api-initial-prompt', default=None, help='API initial prompt')
@click.option('--local-model', default='base', help='Local model name')
@click.option('--local-device', default='auto', help='Local model device setting')
@click.option('--local-compute-type', default='auto', help='Local model compute type')
@click.option('--local-language', default=None, help='Local model language setting')
@click.option('--local-temperature', default=0.0, type=float, help='Local model temperature setting')
@click.option('--local-initial-prompt', default=None, help='Local model initial prompt')
@click.option('--local-condition-on-previous-text/--no-local-condition-on-previous-text', default=True,
              help='Local model condition on previous text')
@click.option('--local-vad-filter/--no-local-vad-filter', default=False, help='Local model VAD filter')
@click.option('--activation-key', default='ctrl+shift+space', help='Activation key combination')
@click.option('--recording-mode', type=click.Choice([mode.value for mode in RecordingMode]),
              default='voice_activity_detection', help='Recording mode')
@click.option('--sound-device', default=None, help='Sound device to use')
@click.option('--sample-rate', default=16000, type=int, help='Sample rate for recording')
@click.option('--silence-duration', default=900, type=int, help='Duration of silence before stopping recording')
@click.option('--writing-key-press-delay', default=0.008, type=float, help='Delay between key presses during typing')
@click.option('--noise-on-completion/--no-noise-on-completion', default=False, help='Play noise on completion')
@click.option('--remove-trailing-period/--no-remove-trailing-period', default=True,
              help='Remove trailing period from transcribed text')
@click.option('--add-trailing-space/--no-add-trailing-space', default=False,
              help='Add trailing space to transcribed text')
@click.option('--remove-capitalization/--no-remove-capitalization', default=False,
              help='Remove capitalization from transcribed text')
@click.option('--print-to-terminal/--no-print-to-terminal', default=True, help='Print transcribed text to terminal')
@click.option('--hide-status-window/--no-hide-status-window', default=False,
              help='Hide the status window during recording')
def main(use_api, api_model, api_language, api_temperature, api_initial_prompt, local_model, local_device,
         local_compute_type,
         local_language, local_temperature, local_initial_prompt, local_condition_on_previous_text, local_vad_filter,
         activation_key, recording_mode, sound_device, sample_rate, silence_duration, writing_key_press_delay,
         noise_on_completion, remove_trailing_period, add_trailing_space, remove_capitalization, print_to_terminal,
         hide_status_window):
    config = Configuration(
        use_api=use_api,
        api_options=APIOptions(model=api_model, language=api_language, temperature=api_temperature,
                               initial_prompt=api_initial_prompt),
        local_model_options=LocalModelOptions(model=local_model, device=local_device, compute_type=local_compute_type,
                                              language=local_language,
                                              temperature=local_temperature, initial_prompt=local_initial_prompt,
                                              condition_on_previous_text=local_condition_on_previous_text,
                                              vad_filter=local_vad_filter),
        activation_key=activation_key,
        recording_mode=RecordingMode(recording_mode),
        sound_device=sound_device,
        sample_rate=sample_rate,
        silence_duration=silence_duration,
        writing_key_press_delay=writing_key_press_delay,
        noise_on_completion=noise_on_completion,
        remove_trailing_period=remove_trailing_period,
        add_trailing_space=add_trailing_space,
        remove_capitalization=remove_capitalization,
        print_to_terminal=print_to_terminal,
        hide_status_window=hide_status_window
    )

    WhisperWriterCli(config)


if __name__ == "__main__":
    main()
