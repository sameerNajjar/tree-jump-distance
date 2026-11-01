"""Exceptions that mark an unfinished computation. Neither one is a NO answer."""


class Timeout(Exception):
    """The time allowance ran out before the computation finished."""


class CapExceeded(Exception):
    """A candidate-count or candidate-memory allowance was reached."""
