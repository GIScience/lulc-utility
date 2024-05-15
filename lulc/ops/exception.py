class OperatorValidationException(Exception):
    """
    Describes a recoverable failure in the application-operator interaction
    """

    pass


class OperatorInteractionException(Exception):
    """
    Describes a fatal failure in the application-operator interaction
    """

    pass
