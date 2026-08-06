from __future__ import annotations

import json
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any,Mapping


class IncomparableSuites(ValueError):pass


@dataclass(frozen=True)
class MetricDelta:
  baseline:float
  candidate:float
  absolute:float
  relative:float|None


@dataclass(frozen=True)
class EvaluationGate:
  passed:bool
  task_success_preserved:bool
  tool_validity_preserved:bool
  all_baseline_passes_still_pass:bool
  reasons:tuple[str,...]


def _load(value:str|Path|Mapping[str,Any])->dict[str,Any]:
  if isinstance(value,Mapping):return dict(value)
  path=Path(value)
  payload=json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload,dict):raise ValueError("evaluation result must be an object")
  return payload


def _delta(base:float,candidate:float)->dict[str,Any]:
  return asdict(MetricDelta(base,candidate,candidate-base,None if base==0 else (candidate-base)/base))


def _tasks(payload:Mapping[str,Any])->dict[str,Mapping[str,Any]]:
  rows=payload.get("results",())
  if not isinstance(rows,list):raise ValueError("evaluation results must be an array")
  ret={}
  for row in rows:
    if not isinstance(row,Mapping) or not row.get("task_id"):continue
    ret[str(row["task_id"])]=row
  return ret


def compare_evaluations(baseline_value:str|Path|Mapping[str,Any],candidate_value:str|Path|Mapping[str,Any],
                        *,max_task_success_drop:float=0.0,max_tool_validity_drop:float=0.0)->dict[str,Any]:
  baseline,candidate=_load(baseline_value),_load(candidate_value)
  if baseline.get("suite_hash")!=candidate.get("suite_hash"):
    raise IncomparableSuites(f"suite_hash differs: baseline={baseline.get('suite_hash')} candidate={candidate.get('suite_hash')}")
  base_tasks,cand_tasks=_tasks(baseline),_tasks(candidate)
  if set(base_tasks)!=set(cand_tasks):
    raise IncomparableSuites(f"task ids differ: baseline={sorted(base_tasks)} candidate={sorted(cand_tasks)}")

  base_success=float(baseline.get("task_success_rate",0.0));cand_success=float(candidate.get("task_success_rate",0.0))
  base_tools=float(baseline.get("tool_call_validity_rate",0.0));cand_tools=float(candidate.get("tool_call_validity_rate",0.0))
  success_ok=cand_success+max_task_success_drop>=base_success
  tools_ok=cand_tools+max_tool_validity_drop>=base_tools
  regressed=[task_id for task_id,row in base_tasks.items() if bool(row.get("passed")) and not bool(cand_tasks[task_id].get("passed"))]
  reasons=[]
  if not success_ok:reasons.append(f"task success regressed {base_success:.4f}->{cand_success:.4f}")
  if not tools_ok:reasons.append(f"tool validity regressed {base_tools:.4f}->{cand_tools:.4f}")
  if regressed:reasons.append(f"baseline-passing tasks regressed: {regressed}")
  gate=EvaluationGate(not reasons,success_ok,tools_ok,not regressed,tuple(reasons))

  per_task=[]
  for task_id in sorted(base_tasks):
    left,right=base_tasks[task_id],cand_tasks[task_id]
    per_task.append({"task_id":task_id,"baseline_passed":bool(left.get("passed")),"candidate_passed":bool(right.get("passed")),
      "latency_ms":_delta(float(left.get("latency_ms",0.0)),float(right.get("latency_ms",0.0))),
      "baseline_tools":left.get("observed_tools",[]),"candidate_tools":right.get("observed_tools",[]),
      "candidate_failure_reasons":right.get("failure_reasons",[])})

  latency={name:_delta(float(baseline.get(name,0.0)),float(candidate.get(name,0.0)))
           for name in ("p50_latency_ms","p95_latency_ms","mean_latency_ms")}
  return {"suite_name":baseline.get("suite_name"),"suite_hash":baseline.get("suite_hash"),"gate":asdict(gate),
    "quality":{"task_success_rate":_delta(base_success,cand_success),"tool_call_validity_rate":_delta(base_tools,cand_tools)},
    "latency":latency,"per_task":per_task,
    "deployable_speedup":gate.passed and latency["p95_latency_ms"]["absolute"]<0,
    "interpretation":"Latency improvement is deployable only when the frozen quality/tool gates pass."}
