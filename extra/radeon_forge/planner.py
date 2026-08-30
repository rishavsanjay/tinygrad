from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .contracts import WorkloadContract
from .knowledge import RetrievalHit
from .permissions import Action


@dataclass(frozen=True)
class PlannerDecision:
  summary: str
  hypothesis: str
  candidate_family: str
  proposed_actions: tuple[Action, ...]
  benchmark_budget: int
  rationale: str
  unknowns: tuple[str, ...] = ()


class LocalEndpointRequired(ValueError): pass
class PlannerProtocolError(RuntimeError): pass


def validate_local_endpoint(endpoint: str) -> str:
  parsed = urlparse(endpoint)
  if parsed.scheme != "http" or not parsed.hostname: raise LocalEndpointRequired("planner endpoint must be a local HTTP URL")
  hostname = parsed.hostname.lower()
  is_loopback = hostname == "localhost"
  if not is_loopback:
    try: is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError: is_loopback = False
  if not is_loopback: raise LocalEndpointRequired(f"remote planner endpoint is forbidden: {hostname}")
  if parsed.username or parsed.password: raise LocalEndpointRequired("credentials must not be embedded in the planner URL")
  return endpoint.rstrip("/")


class LocalPlanner:
  """OpenAI-compatible planner client restricted to loopback.

  Planner output is advisory structured data. It cannot execute tools, issue
  permissions, or override deterministic oracle results.
  """

  def __init__(self, endpoint: str, model: str, timeout_seconds: int = 120):
    self.endpoint = validate_local_endpoint(endpoint)
    self.model = model
    self.timeout_seconds = timeout_seconds

  @staticmethod
  def _evidence(hits: Sequence[RetrievalHit]) -> list[dict[str, Any]]:
    return [{"citation": hit.citation, "score": hit.score, "text": hit.chunk.text} for hit in hits]

  def plan(self, request: str, contract: WorkloadContract, evidence: Sequence[RetrievalHit] = (),
           memory: Sequence[Mapping[str, Any]] = ()) -> PlannerDecision:
    system = """You are Radeon Forge, a private AMD performance-engineering planner.
Return one JSON object only. Propose falsifiable hypotheses and bounded tool actions.
Never claim a speedup before hardware measurement. Never bypass correctness, quality,
privacy, stability, resource, or permission gates. Generated code is disposable; the
contract and oracle are authoritative."""
    user = {
      "request": request,
      "workload_contract": {
        "name": contract.name,
        "target": contract.target,
        "objective": contract.objective.value,
        "max_abs_error": contract.max_abs_error,
        "max_rel_error": contract.max_rel_error,
        "max_vgprs": contract.max_vgprs,
        "max_lds_bytes": contract.max_lds_bytes,
        "forbid_spills": contract.forbid_spills,
        "metadata": dict(contract.metadata),
      },
      "local_evidence": self._evidence(evidence),
      "recent_memory": list(memory),
      "required_schema": {
        "summary": "string",
        "hypothesis": "string",
        "candidate_family": "string",
        "proposed_actions": [action.value for action in Action],
        "benchmark_budget": "positive integer",
        "rationale": "string",
        "unknowns": ["string"],
      },
    }
    payload = {
      "model": self.model,
      "temperature": 0,
      "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, sort_keys=True)}],
      "response_format": {"type": "json_object"},
    }
    request_obj = urllib.request.Request(
      self.endpoint + "/chat/completions",
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    try:
      with urllib.request.urlopen(request_obj, timeout=self.timeout_seconds) as response:
        result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
      raise PlannerProtocolError(f"local planner request failed: {exc}") from exc
    try:
      content = result["choices"][0]["message"]["content"]
      data = json.loads(content)
      actions = tuple(Action(value) for value in data["proposed_actions"])
      budget = int(data["benchmark_budget"])
      if budget <= 0: raise ValueError("benchmark_budget must be positive")
      return PlannerDecision(
        summary=str(data["summary"]),
        hypothesis=str(data["hypothesis"]),
        candidate_family=str(data["candidate_family"]),
        proposed_actions=actions,
        benchmark_budget=budget,
        rationale=str(data["rationale"]),
        unknowns=tuple(str(value) for value in data.get("unknowns", ())),
      )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
      raise PlannerProtocolError(f"invalid planner response: {exc}") from exc
