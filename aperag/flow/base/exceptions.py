class FlowError(Exception):
    """Base exception for flow engine"""
    pass

class CycleError(FlowError):
    """Raised when a cycle is detected in the flow graph"""
    pass

class TypeMismatchError(FlowError):
    """Raised when input/output types don't match"""
    pass

class NodeNotFoundError(FlowError):
    """Raised when a node is not found"""
    pass

class ValidationError(FlowError):
    """Raised when input validation fails"""
    pass