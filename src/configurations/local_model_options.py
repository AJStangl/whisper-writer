from typing import Optional

class LocalModelOptions:
    def __init__(self, model: str = 'base', device: str = 'auto', compute_type: str = 'auto',
                 language: Optional[str] = None,
                 temperature: float = 0.0, initial_prompt: Optional[str] = None,
                 condition_on_previous_text: bool = True, vad_filter: bool = False):
        self.model = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.temperature = temperature
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
