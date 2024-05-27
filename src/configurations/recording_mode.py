from enum import Enum


class RecordingMode(Enum):
    VOICE_ACTIVITY_DETECTION = 'voice_activity_detection'
    PRESS_TO_TOGGLE = 'press_to_toggle'
    HOLD_TO_RECORD = 'hold_to_record'
    CONTINUOUS = 'continuous'
