from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from .contracts import WorkloadContract
from .knowledge import LocalKnowledgeBase, RetrievalHit
from .ledger import ExperimentLedger
from .permissions import Action, PermissionController, PermissionGrant
from .planner import PlannerDecision


class AgentState(str, Enum):
  READY = "ready"
  AWAITING_APPROVAL = "awaiting_approval"
  APPROVED = "approved"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"


class AgentStateError(RuntimeError): pass


class PlannerLike(Protocol):
  def plan(self, request: str, contract: WorkloadContract, evidence: Sequence[RetrievalHit] = (),
           memory: Sequence[Mapping[str, Any]] = ()) -> PlannerDecision: ...


@dataclass(frozen=True)
class Proposal:
  proposal_id: str
  user_request: str
  decision: PlannerDecision
  evidence_citations: tuple[str, ...]
  contract: WorkloadContract
  supersedes: str | None = None


@dataclass(frozen=True)
class ApprovalReceipt:
  proposal_id: str
  approved_actions: tuple[Action, ...]
  user_reason: str
  grant: PermissionGrant


class ForgeAgent:
  """Stateful multi-turn coordinator for planning and permissioned execution.

  This object does not execute kernels by itself. It coordinates the local
  planner, retrieval, memory, and user approval boundary. Deterministic tools
  must call `authorize` immediately before each consequential action.
  """

  def __init__(self, contract: WorkloadContract, planner: PlannerLike, knowledge: LocalKnowledgeBase,
               ledger: ExperimentLedger, permissions: PermissionController | None = None):
    self.contract = contract
    self.planner = planner
    self.knowledge = knowledge
    self.ledger = ledger
    self.permissions = permissions or PermissionController()
    self.state = AgentState.READY
    self.current_proposal: Proposal | None = None
    self.approval: ApprovalReceipt | None = None
    self._conversation: list[dict[str, Any]] = []

  def _memory(self, limit: int = 20) -> tuple[Mapping[str, Any], ...]:
    records = tuple(self.ledger.records())
    return tuple(records[-limit:])

  def propose(self, user_request: str, top_k: int = 5) -> Proposal:
    if self.state is AgentState.RUNNING: raise AgentStateError("cannot create a new proposal while tools are running")
    request = user_request.strip()
    if not request: raise ValueError("user_request must not be empty")
    hits = self.knowledge.search(request, top_k=top_k)
    decision = self.planner.plan(request, self.contract, evidence=hits, memory=self._memory())
    previous = self.current_proposal.proposal_id if self.current_proposal is not None else None
    proposal = Proposal(secrets.token_urlsafe(12), request, decision, tuple(hit.citation for hit in hits), self.contract, previous)
    self.current_proposal = proposal
    self.approval = None
    self.state = AgentState.AWAITING_APPROVAL
    self._conversation.append({"role": "user", "content": request})
    self._conversation.append({"role": "agent", "proposal_id": proposal.proposal_id, "decision": asdict(decision)})
    self.ledger.append("proposal_created", proposal)
    return proposal

  def revise(self, user_request: str, contract_changes: Mapping[str, Any] | None = None) -> Proposal:
    if contract_changes:
      allowed = {field for field in self.contract.__dataclass_fields__ if field not in {"name", "target"}}
      unknown = set(contract_changes) - allowed
      if unknown: raise ValueError(f"unsupported contract fields: {sorted(unknown)}")
      self.contract = replace(self.contract, **dict(contract_changes))
      self.ledger.append("contract_revised", {"changes": dict(contract_changes), "contract": self.contract})
    return self.propose(user_request)

  def approve(self, proposal_id: str, user_reason: str, max_tool_uses: int | None = None) -> ApprovalReceipt:
    if self.state is not AgentState.AWAITING_APPROVAL or self.current_proposal is None:
      raise AgentStateError("there is no proposal awaiting approval")
    if proposal_id != self.current_proposal.proposal_id: raise AgentStateError("approval does not match the current proposal")
    actions = self.current_proposal.decision.proposed_actions
    if not actions: raise AgentStateError("proposal contains no executable actions")
    reason = user_reason.strip()
    if not reason: raise ValueError("approval requires a user reason")
    uses = max_tool_uses if max_tool_uses is not None else max(1, self.current_proposal.decision.benchmark_budget + len(actions))
    grant = self.permissions.issue(actions, reason, max_uses=uses)
    receipt = ApprovalReceipt(proposal_id, actions, reason, grant)
    self.approval = receipt
    self.state = AgentState.APPROVED
    # Never persist the bearer token. The receipt fields excluding the grant are
    # enough to audit what the user approved.
    self.ledger.append("proposal_approved", {"proposal_id": proposal_id, "approved_actions": [action.value for action in actions],
                                             "user_reason": reason, "max_tool_uses": uses})
    return receipt

  def authorize(self, action: Action) -> PermissionGrant:
    if self.state not in {AgentState.APPROVED, AgentState.RUNNING} or self.approval is None:
      raise AgentStateError(f"action {action.value} is not approved")
    grant = self.permissions.authorize(self.approval.grant.token, action)
    self.state = AgentState.RUNNING
    self.ledger.append("tool_authorized", {"proposal_id": self.approval.proposal_id, "action": action.value, "reason": grant.reason})
    return grant

  def complete(self, result: Mapping[str, Any]) -> None:
    if self.state not in {AgentState.APPROVED, AgentState.RUNNING}: raise AgentStateError("no approved execution is active")
    self.state = AgentState.COMPLETED
    self.ledger.append("execution_completed", {"proposal_id": self.approval.proposal_id if self.approval else None, "result": dict(result)})

  def fail(self, reason: str, evidence: Mapping[str, Any] | None = None) -> None:
    if self.state not in {AgentState.APPROVED, AgentState.RUNNING}: raise AgentStateError("no approved execution is active")
    self.state = AgentState.FAILED
    self.ledger.append("execution_failed", {"proposal_id": self.approval.proposal_id if self.approval else None,
                                             "reason": reason, "evidence": dict(evidence or {})})

  @property
  def conversation(self) -> tuple[Mapping[str, Any], ...]:
    return tuple(self._conversation)
