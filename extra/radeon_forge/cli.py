from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .command_backend import CommandHarness
from .contracts import Objective, WorkloadContract
from .families import rdna3_asm_matmul_family
from .ledger import ExperimentLedger
from .permissions import Action, PermissionController
from .tuner import successive_halving


def _parse_budgets(value: str) -> tuple[int, ...]:
  budgets = tuple(int(part) for part in value.split(",") if part.strip())
  if not budgets or any(budget <= 0 for budget in budgets): raise argparse.ArgumentTypeError("budgets must be positive comma-separated integers")
  return budgets


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Oracle-guided Radeon Forge autotuner")
  parser.add_argument("--root", default=".", help="tinygrad repository root")
  parser.add_argument("--family", choices=("rdna3-asm-matmul",), default="rdna3-asm-matmul")
  parser.add_argument("--n", type=int, default=1024, help="square GEMM dimension for the first gfx1100 hardware family")
  parser.add_argument("--budgets", type=_parse_budgets, default=(3, 10, 30), help="successive-halving trial counts, e.g. 3,10,30")
  parser.add_argument("--reduction", type=int, default=4)
  parser.add_argument("--ledger", default="evidence/radeon_forge/events.jsonl")
  parser.add_argument("--dry-run", action="store_true", help="print the plan without compiling or benchmarking")
  parser.add_argument("--approve-benchmark", action="store_true", help="explicitly approve local W7900 benchmark execution")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  root = Path(args.root).resolve()
  family = rdna3_asm_matmul_family(root, args.n)
  candidates = family.candidates()
  plan = {
    "family": family.name,
    "candidate_count": len(candidates),
    "budgets": args.budgets,
    "reduction": args.reduction,
    "objective": Objective.P95_LATENCY_US.value,
    "target": "gfx1100",
    "reference": "tinygrad matmul",
    "candidate_ids": [candidate.candidate_id for candidate in candidates],
  }
  print(json.dumps({"plan": plan}, indent=2, sort_keys=True))
  if args.dry_run: return 0
  if not args.approve_benchmark:
    print("Refusing hardware execution: pass --approve-benchmark after reviewing the plan.", file=sys.stderr)
    return 2

  ledger = ExperimentLedger(root / args.ledger)
  ledger.append("optimization_requested", plan)
  controller = PermissionController()
  # Upper bound: every candidate in every round. The actual tuner consumes fewer
  # uses after successive halving removes candidates.
  grant = controller.issue([Action.BENCHMARK], f"approved benchmark plan for {family.name}", max_uses=len(candidates) * len(args.budgets))
  harness = CommandHarness([sys.executable, "-m", "extra.radeon_forge.workloads.rdna3_asm_matmul"], cwd=root)

  def approved_evaluator(candidate, budget):
    authorized = controller.authorize(grant.token, Action.BENCHMARK)
    ledger.append("permission_used", {"action": Action.BENCHMARK.value, "reason": authorized.reason,
                                      "candidate_id": candidate.candidate_id, "budget": budget})
    return harness(candidate, budget)

  contract = WorkloadContract(
    name="rdna3-asm-matmul",
    target="gfx1100",
    objective=Objective.P95_LATENCY_US,
    max_abs_error=0.1,
    max_rel_error=1000.0,
    max_vgprs=192,
    max_lds_bytes=32768,
    forbid_spills=True,
    metadata={"final_submission_metric": False, "purpose": "validate the Forge hardware tuning loop"},
  )
  ledger.append("workload_contract", contract)
  summary = successive_halving(candidates, approved_evaluator, contract, budgets=args.budgets, reduction=args.reduction, ledger=ledger)
  if summary.winner is None:
    print(json.dumps({"status": "no_feasible_candidate", "evaluations": len(summary.evaluated)}, indent=2))
    return 1
  print(json.dumps({"status": "winner", "rounds": summary.rounds, "evaluations": len(summary.evaluated),
                    "result": summary.winner.to_dict()}, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
