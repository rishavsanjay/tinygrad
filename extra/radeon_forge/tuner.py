from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil

from .contracts import Candidate, TrialResult, WorkloadContract
from .ledger import ExperimentLedger


Evaluator = Callable[[Candidate, int], TrialResult]


@dataclass(frozen=True)
class TuningSummary:
  winner: TrialResult | None
  evaluated: tuple[TrialResult, ...]
  rounds: int


def _rank(results: Sequence[TrialResult], contract: WorkloadContract) -> list[TrialResult]:
  return sorted(results, key=lambda result: (not result.is_feasible(contract), result.metric(contract.objective), result.candidate.candidate_id))


def successive_halving(candidates: Sequence[Candidate], evaluator: Evaluator, contract: WorkloadContract, budgets: Sequence[int] = (5, 20, 100),
                       reduction: int = 4, ledger: ExperimentLedger | None = None) -> TuningSummary:
  """Evaluate candidates with increasing benchmark budgets.

  Correctness and resource constraints are hard gates. A faster candidate can
  never outrank a candidate that passes the workload contract.
  """
  if not candidates: return TuningSummary(None, (), 0)
  if not budgets or any(budget <= 0 for budget in budgets): raise ValueError("budgets must contain positive iteration counts")
  if reduction < 2: raise ValueError("reduction must be at least 2")

  active = list(candidates)
  all_results: list[TrialResult] = []
  completed_rounds = 0
  for round_index, budget in enumerate(budgets):
    completed_rounds += 1
    round_results: list[TrialResult] = []
    for candidate in active:
      if ledger is not None: ledger.append("trial_started", {"round": round_index, "budget": budget, "candidate": candidate})
      try:
        result = evaluator(candidate, budget)
      except Exception as exc:  # the optimizer records failures rather than silently losing candidates
        from .contracts import CorrectnessReport
        result = TrialResult(candidate=candidate, correctness=CorrectnessReport(False, reason=f"evaluator exception: {exc!r}"),
                             compile_ok=False, stable=False, rejected_reason="evaluator_exception")
      round_results.append(result)
      all_results.append(result)
      if ledger is not None: ledger.append("trial_finished", {"round": round_index, "budget": budget, "result": result})

    ranked = _rank(round_results, contract)
    feasible = [result for result in ranked if result.is_feasible(contract)]
    if not feasible:
      if ledger is not None: ledger.append("round_failed", {"round": round_index, "reason": "no feasible candidates"})
      return TuningSummary(None, tuple(all_results), completed_rounds)

    if round_index == len(budgets) - 1:
      winner = feasible[0]
      if ledger is not None: ledger.append("winner_selected", winner)
      return TuningSummary(winner, tuple(all_results), completed_rounds)

    keep = max(1, ceil(len(feasible) / reduction))
    active = [result.candidate for result in feasible[:keep]]
    if ledger is not None: ledger.append("round_survivors", {"round": round_index, "candidate_ids": [candidate.candidate_id for candidate in active]})

  raise AssertionError("unreachable")
