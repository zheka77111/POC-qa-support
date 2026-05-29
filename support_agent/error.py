

class SupportAgentNodeError(Exception):
    """Base exception for support graph node-level errors."""


class ReceiveRequestNodeError(SupportAgentNodeError):
    pass


class ClassifyComplaintNodeError(SupportAgentNodeError):
    pass


class ClassifyUrgencyNodeError(SupportAgentNodeError):
    pass


class ClassifyCategoryNodeError(SupportAgentNodeError):
    pass


class GatherClassificationsNodeError(SupportAgentNodeError):
    pass


class EscalateComplaintNodeError(SupportAgentNodeError):
    pass


class DialogAgentNodeError(SupportAgentNodeError):
    pass


class FinalizeResponseNodeError(SupportAgentNodeError):
    pass


class KnowledgeBaseSearchToolError(DialogAgentNodeError):
    pass
