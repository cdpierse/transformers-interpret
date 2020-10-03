class AttributionTypeNotSupportedError(RuntimeError):
    "Raised when a particular attribution type is not yet supported by an explainer"


class AttributionsNotCalculatedError(RuntimeError):
    "Raised when a user attempts to access the attributions for a model and sequence before they have be been summarized"


class InputIdsNotCalculatedError(RuntimeError):
    "Raised when a user attempts to call a method or attribute that requires input ids"
