class OperatorValidationError(Exception):
    """
    Describes a recoverable failure in the application-operator interaction
    """

    pass


class OperatorInteractionError(Exception):
    """
    Describes a fatal failure in the application-operator interaction
    """

    pass
