from __future__ import annotations

import argparse, json, sys
from pathlib import Path

from ..backends.plugins import default_worker_plugins
from ..runtime import ForgeEngine, JsonlProcessBackend
from .suite import AgentTaskSuite, run_suite


def main() -> None:
  parser=argparse.ArgumentParser(description="Run a frozen private-agent suite on the local Radeon Forge engine")
  parser.add_argument("--suite",type=Path,required=True)
  parser.add_argument("--workspace",type=Path,default=Path.cwd())
  parser.add_argument("--model-plugin",default="legacy-llama")
  parser.add_argument("--model",type=Path,required=True)
  parser.add_argument("--tokenizer",type=Path)
  parser.add_argument("--size",default="1B")
  parser.add_argument("--quantize")
  parser.add_argument("--max-context",type=int,default=8192)
  parser.add_argument("--seed",type=int,default=42)
  parser.add_argument("--output",type=Path)
  args=parser.parse_args()

  plugin=default_worker_plugins().get(args.model_plugin)
  command=plugin.command({"model":args.model,"tokenizer":args.tokenizer,"size":args.size,"quantize":args.quantize,
                          "max_context":args.max_context,"seed":args.seed})
  engine=ForgeEngine(JsonlProcessBackend(command),args.workspace)
  try:result=run_suite(engine,AgentTaskSuite.load(args.suite)).to_dict()
  finally:engine.close()
  text=json.dumps(result,indent=2,default=str)+"\n"
  if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding="utf-8")
  sys.stdout.write(text)
  raise SystemExit(0 if result["passed_tasks"]==result["total_tasks"] else 2)


if __name__=="__main__":main()
