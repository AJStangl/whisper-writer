from typing import Optional, Dict, Any
from src.configurations.api_options import APIOptions
from src.configurations.local_model_options import LocalModelOptions
from src.configurations.recording_mode import RecordingMode


class Configuration:
    def __init__(self, use_api: bool = False, api_options: Optional[APIOptions] = None,
                 local_model_options: Optional[LocalModelOptions] = None,
                 activation_key: str = 'ctrl+shift+space',
                 recording_mode: RecordingMode = RecordingMode.VOICE_ACTIVITY_DETECTION,
                 sound_device: Optional[str] = None,
                 sample_rate: int = 16000, silence_duration: int = 900, writing_key_press_delay: float = 0.008,
                 noise_on_completion: bool = False,
                 remove_trailing_period: bool = True, add_trailing_space: bool = False,
                 remove_capitalization: bool = False, print_to_terminal: bool = True,
                 hide_status_window: bool = False):
        self.use_api = use_api
        self.api_options = api_options if api_options else APIOptions()
        self.local_model_options = local_model_options if local_model_options else LocalModelOptions()
        self.activation_key = activation_key
        self.recording_mode = recording_mode
        self.sound_device = sound_device
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.writing_key_press_delay = writing_key_press_delay
        self.noise_on_completion = noise_on_completion
        self.remove_trailing_period = remove_trailing_period
        self.add_trailing_space = add_trailing_space
        self.remove_capitalization = remove_capitalization
        self.print_to_terminal = print_to_terminal
        self.hide_status_window = hide_status_window

        self.validate_configuration()

    @classmethod

    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Configuration':
        return cls(
            use_api=config_dict.get('use_api', False),
            api_options=APIOptions(**config_dict.get('api_options', {})),
            local_model_options=LocalModelOptions(**config_dict.get('local_model_options', {})),
            activation_key=config_dict.get('activation_key', 'ctrl+shift+space'),
            recording_mode=RecordingMode(config_dict.get('recording_mode', 'voice_activity_detection')),
            sound_device=config_dict.get('sound_device'),
            sample_rate=config_dict.get('sample_rate', 16000),
            silence_duration=config_dict.get('silence_duration', 900),
            writing_key_press_delay=config_dict.get('writing_key_press_delay', 0.008),
            noise_on_completion=config_dict.get('noise_on_completion', False),
            remove_trailing_period=config_dict.get('remove_trailing_period', True),
            add_trailing_space=config_dict.get('add_trailing_space', False),
            remove_capitalization=config_dict.get('remove_capitalization', False),
            print_to_terminal=config_dict.get('print_to_terminal', True),
            hide_status_window=config_dict.get('hide_status_window', False)
        )

    def validate_configuration(self) -> None:
        if not isinstance(self.recording_mode, RecordingMode):
            raise ValueError(f"Invalid recording mode: {self.recording_mode}")
