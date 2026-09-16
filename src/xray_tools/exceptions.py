"""Exceptions raised by xray-tools workflow helpers."""


class CIAOWorkflowError(RuntimeError):
    """Base error for an unavailable or unusable CIAO environment."""
