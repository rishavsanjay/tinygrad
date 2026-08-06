from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable


class Action(str, Enum):
  INSPECT = "inspect"
  COMPILE = "compile"
  BENCHMARK = "benchmark"
  WRITE_GENERATED_SOURCE = "write_generated_source"
  RESTART_LOCAL_SERVICE = "restart_local_service"
  DEPLOY = "deploy"
  DOWNLOAD_MODEL = "download_model"


@dataclass(frozen=True)
class PermissionGrant:
  token: str
  actions: frozenset[Action]
  reason: str
  issued_at_utc: datetime
  expires_at_utc: datetime
  max_uses: int


class PermissionDenied(RuntimeError): pass


class PermissionController:
  """Deny-by-default session permission controller.

  Grants are explicit, scoped, expiring, and usage-limited. The controller is
  intentionally independent of the language model; an agent cannot authorize
  its own consequential action.
  """

  def __init__(self):
    self._grants: dict[str, PermissionGrant] = {}
    self._uses: dict[str, int] = {}

  def issue(self, actions: Iterable[Action], reason: str, ttl_seconds: int = 900, max_uses: int = 1) -> PermissionGrant:
    scope = frozenset(actions)
    if not scope: raise ValueError("a permission grant must include at least one action")
    if not reason.strip(): raise ValueError("a permission grant requires a reason")
    if ttl_seconds <= 0 or max_uses <= 0: raise ValueError("ttl_seconds and max_uses must be positive")
    now = datetime.now(timezone.utc)
    grant = PermissionGrant(secrets.token_urlsafe(24), scope, reason.strip(), now, now + timedelta(seconds=ttl_seconds), max_uses)
    self._grants[grant.token] = grant
    self._uses[grant.token] = 0
    return grant

  def revoke(self, token: str) -> None:
    self._grants.pop(token, None)
    self._uses.pop(token, None)

  def authorize(self, token: str | None, action: Action) -> PermissionGrant:
    if token is None or token not in self._grants: raise PermissionDenied(f"no active grant for {action.value}")
    grant = self._grants[token]
    if datetime.now(timezone.utc) >= grant.expires_at_utc:
      self.revoke(token)
      raise PermissionDenied(f"grant expired for {action.value}")
    if action not in grant.actions: raise PermissionDenied(f"grant does not include {action.value}")
    uses = self._uses[token]
    if uses >= grant.max_uses:
      self.revoke(token)
      raise PermissionDenied(f"grant exhausted for {action.value}")
    self._uses[token] = uses + 1
    if self._uses[token] >= grant.max_uses:
      # The returned immutable grant remains valid as evidence, but the token
      # cannot be reused for a second action.
      self._grants.pop(token, None)
    return grant
