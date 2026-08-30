from __future__ import annotations

import hashlib, json, math, re, statistics, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..runtime import ForgeEngine
from ..runtime.session import SessionState


@dataclass(frozen=True)
class AgentTask:
  task_id: str
  prompt: str
  expected_tools: tuple[str, ...] = ()
  allowed_tools: tuple[str, ...] = ()
  required_output_regex: tuple[str, ...] = ()
  forbidden_output_regex: tuple[str, ...] = ()
  max_tokens: int = 256
  max_agent_steps: int = 8
  timeout_seconds: float = 120.0
  metadata: Mapping[str, Any] = field(default_factory=dict)

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any]) -> AgentTask:
    task_id, prompt = str(value.get("task_id", "")).strip(), str(value.get("prompt", "")).strip()
    if not task_id or not prompt: raise ValueError("task_id and prompt are required")
    def strings(name: str) -> tuple[str, ...]:
      raw=value.get(name,())
      if not isinstance(raw,list) or not all(isinstance(item,str) for item in raw): raise ValueError(f"{name} must be a string array")
      return tuple(raw)
    return cls(task_id,prompt,strings("expected_tools"),strings("allowed_tools"),strings("required_output_regex"),
               strings("forbidden_output_regex"),int(value.get("max_tokens",256)),int(value.get("max_agent_steps",8)),
               float(value.get("timeout_seconds",120.0)),dict(value.get("metadata",{})))


@dataclass(frozen=True)
class AgentTaskSuite:
  name: str
  tasks: tuple[AgentTask, ...]
  description: str = ""
  metadata: Mapping[str, Any] = field(default_factory=dict)

  @property
  def suite_hash(self) -> str:
    payload=asdict(self)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()

  @classmethod
  def load(cls,path: str|Path) -> AgentTaskSuite:
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("format_version",0)) != 1: raise ValueError("unsupported agent suite format")
    tasks=tuple(AgentTask.from_mapping(item) for item in payload.get("tasks",()))
    if not tasks: raise ValueError("suite requires at least one task")
    identifiers=[task.task_id for task in tasks]
    if len(identifiers)!=len(set(identifiers)): raise ValueError("task ids must be unique")
    return cls(str(payload.get("name","")).strip() or Path(path).stem,tasks,str(payload.get("description","")),dict(payload.get("metadata",{})))


@dataclass(frozen=True)
class TaskResult:
  task_id: str
  passed: bool
  latency_ms: float
  final_output: str
  observed_tools: tuple[str, ...]
  expected_tools: tuple[str, ...]
  tool_call_valid: bool
  output_valid: bool
  terminal_state: str
  generated_tokens: int
  prompt_tokens: int
  failure_reasons: tuple[str, ...]
  session_id: str
  trace_id: str


@dataclass(frozen=True)
class SuiteResult:
  suite_name: str
  suite_hash: str
  passed_tasks: int
  total_tasks: int
  task_success_rate: float
  tool_call_validity_rate: float
  p50_latency_ms: float
  p95_latency_ms: float
  mean_latency_ms: float
  results: tuple[TaskResult, ...]

  def to_dict(self) -> dict[str, Any]: return asdict(self)


def _percentile(values: Sequence[float], fraction: float) -> float:
  if not values:return 0.0
  ordered=sorted(values); index=max(0,min(len(ordered)-1,math.ceil(len(ordered)*fraction)-1))
  return ordered[index]


def _last_assistant(messages: Sequence[Mapping[str, Any]]) -> str:
  for message in reversed(messages):
    if message.get("role")=="assistant" and "<tool_call>" not in str(message.get("content","")):
      return str(message.get("content",""))
  return ""


def _tool_sequence(session) -> tuple[str, ...]:
  names=[]
  for event in session.events:
    if event.kind=="tool_started":
      call=event.data.get("call",{})
      if isinstance(call,Mapping) and call.get("name"):names.append(str(call["name"]))
  return tuple(names)


def _token_usage(session) -> tuple[int,int]:
  prompt=generated=0
  for event in session.events:
    if event.kind=="prefill":prompt=max(prompt,int(event.data.get("prompt_tokens",0)))
    elif event.kind=="generation_done":generated+=int((event.data.get("metrics",{}) or {}).get("generated_tokens",0))
  return prompt,generated


def run_task(engine: ForgeEngine,task: AgentTask) -> TaskResult:
  session=engine.create_session(); session.max_agent_steps=task.max_agent_steps
  started=time.perf_counter_ns(); deadline=time.monotonic()+task.timeout_seconds
  failures=[]
  try: engine.run_message(session.session_id,task.prompt,task.max_tokens,0.0)
  except Exception as exc: failures.append(f"initial generation failed: {type(exc).__name__}: {exc}")
  while session.state is SessionState.AWAITING_TOOL_APPROVAL and not failures:
    if time.monotonic()>=deadline:
      failures.append("task timeout while awaiting tool execution");break
    call=session.pending_tool_call
    if call is None:
      failures.append("session requested approval without a pending tool");break
    if call.name not in task.allowed_tools:
      failures.append(f"tool {call.name!r} is not allowed by the frozen task policy")
      session.reject_tool("not allowed by frozen evaluation policy")
      break
    try:engine.run_tool_approval(session.session_id,f"Frozen suite permits local tool {call.name}",task.max_tokens)
    except Exception as exc:
      failures.append(f"tool/resume failed: {type(exc).__name__}: {exc}");break
  latency_ms=(time.perf_counter_ns()-started)/1e6
  observed=_tool_sequence(session)
  tool_valid=observed==task.expected_tools and all(name in task.allowed_tools for name in observed)
  if not tool_valid:failures.append(f"tool sequence expected={task.expected_tools!r} observed={observed!r}")
  output=_last_assistant(session.messages)
  output_valid=True
  for pattern in task.required_output_regex:
    if re.search(pattern,output,re.I|re.S) is None:
      output_valid=False;failures.append(f"required output pattern did not match: {pattern!r}")
  for pattern in task.forbidden_output_regex:
    if re.search(pattern,output,re.I|re.S) is not None:
      output_valid=False;failures.append(f"forbidden output pattern matched: {pattern!r}")
  if session.state not in {SessionState.COMPLETED,SessionState.CANCELLED}:
    failures.append(f"non-terminal-success session state: {session.state.value}")
  if latency_ms>task.timeout_seconds*1000:failures.append("task exceeded latency timeout")
  prompt_tokens,generated_tokens=_token_usage(session)
  passed=not failures and tool_valid and output_valid and session.state is SessionState.COMPLETED
  return TaskResult(task.task_id,passed,latency_ms,output,observed,task.expected_tools,tool_valid,output_valid,
                    session.state.value,generated_tokens,prompt_tokens,tuple(failures),session.session_id,session.trace.trace_id)


def run_suite(engine: ForgeEngine,suite: AgentTaskSuite) -> SuiteResult:
  results=tuple(run_task(engine,task) for task in suite.tasks)
  latencies=[item.latency_ms for item in results]
  return SuiteResult(suite.name,suite.suite_hash,sum(item.passed for item in results),len(results),
    sum(item.passed for item in results)/len(results),sum(item.tool_call_valid for item in results)/len(results),
    statistics.median(latencies),_percentile(latencies,0.95),statistics.mean(latencies),results)
