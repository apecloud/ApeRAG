'use client';

export type {
  ActivityData,
  ActivityIntent,
  AgentCitationPart,
  AgentElicitationPart,
  AgentMessagePart,
  AgentSourceDocumentPart,
  AgentSourceUrlPart,
  AgentStreamStatus,
  AgentTextPart,
  AgentToolConsentPart,
  AgentToolPart,
  CitationData,
  CitationLocation,
  ElicitationData,
  StreamPart,
  ToolConsentData,
  ToolConsentRisk,
  UserActivityEnvelope,
} from './types';

export {
  AI_SDK_V5_HEADER,
  AI_SDK_V5_HEADER_VALUE,
} from './types';

export type {
  AgentArtifactEnvelope,
  AgentRuntimeKind,
  AgentTimelineEventEnvelope,
  AgentTurnCreateInput,
  AgentTurnCreateResponse,
  AgentTurnEnvelope,
  AgentTurnSnapshotEnvelope,
} from './api';

export {
  cancelAgentTurn,
  createAgentTurn,
  decideToolConsent,
  getAgentArtifact,
  getAgentTurnSnapshot,
  submitElicitation,
} from './api';

export {
  useAgentTurnStream,
  type UseAgentTurnStreamInput,
  type UseAgentTurnStreamResult,
} from './use-agent-turn-stream';

export {
  isTerminalBackendStatus,
  mapBackendTurnStatus,
} from './snapshot-fallback';
