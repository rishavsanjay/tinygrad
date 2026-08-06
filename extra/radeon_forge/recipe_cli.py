from __future__ import annotations

import argparse, json
from dataclasses import asdict
from pathlib import Path

from .synthesis import CandidateWorkspace, ForgeRecipe, RecipeLibrary, export_recipe


def main() -> None:
  parser = argparse.ArgumentParser(description="Import, inspect and export portable Radeon Forge optimization recipes")
  sub = parser.add_subparsers(dest="command", required=True)

  inspect_p = sub.add_parser("inspect", help="Validate and inspect one .forge.toml file without installing it")
  inspect_p.add_argument("recipe", type=Path)

  install_p = sub.add_parser("install", help="Install a recipe into a local Forge optimization library")
  install_p.add_argument("recipe", type=Path)
  install_p.add_argument("--workspace", type=Path, default=Path.home() / ".cache" / "radeon-forge")

  list_p = sub.add_parser("list", help="List installed portable recipes")
  list_p.add_argument("--workspace", type=Path, default=Path.home() / ".cache" / "radeon-forge")

  export_p = sub.add_parser("export", help="Export a local contract and optional implementation cache as one file")
  export_p.add_argument("--workspace", type=Path, default=Path.home() / ".cache" / "radeon-forge")
  export_p.add_argument("--spec-id", required=True)
  export_p.add_argument("--candidate-id")
  export_p.add_argument("--output", type=Path, required=True)

  args = parser.parse_args()
  if args.command == "inspect":
    recipe = ForgeRecipe.load(args.recipe)
    payload = asdict(recipe)
    payload.pop("source_text", None)
    payload["recipe_id"] = recipe.recipe_id
    payload["artifacts"] = [{**asdict(x), "content": f"<{len(x.content.encode())} bytes>", "sha256": x.sha256} for x in recipe.artifacts]
    print(json.dumps(payload, indent=2, default=str))
    return

  workspace = CandidateWorkspace(args.workspace)
  library = RecipeLibrary(workspace)
  if args.command == "install": print(json.dumps(asdict(library.install_file(args.recipe)), indent=2))
  elif args.command == "list": print(json.dumps([asdict(x) for x in library.installed()], indent=2))
  elif args.command == "export":
    path = export_recipe(workspace, args.spec_id, args.output, args.candidate_id)
    print(json.dumps({"path": str(path.resolve()), "bytes": path.stat().st_size}, indent=2))


if __name__ == "__main__": main()
