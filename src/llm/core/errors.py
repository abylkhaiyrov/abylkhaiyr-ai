class LLMError(Exception):
    exit_code = 1


class ConfigError(LLMError):
    exit_code = 2


class HardwareError(LLMError):
    exit_code = 3


class DataError(LLMError):
    exit_code = 4


class TokenizerError(LLMError):
    exit_code = 5


class CheckpointError(LLMError):
    exit_code = 6


class TrainingError(LLMError):
    exit_code = 7


class ResourceError(LLMError):
    exit_code = 8


class SandboxError(LLMError):
    exit_code = 9
