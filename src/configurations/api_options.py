from typing import Optional

class APIOptions:
    def __init__(self, model: str = 'whisper-1', language: Optional[str] = None, temperature: float = 0.0,
                 initial_prompt: Optional[str] = None):
        self.model = model
        self.language = language
        self.temperature = temperature
        self.initial_prompt = initial_prompt

