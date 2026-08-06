from __future__ import annotations

import argparse, json, mimetypes, re, sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..backends.fake import ScriptedBackend
from ..runtime import ForgeEngine, JsonlProcessBackend


class ForgeRequestHandler(BaseHTTPRequestHandler):
  engine: ForgeEngine
  static_root = Path(__file__).parent / "static"

  def log_message(self, format, *args): print(f"[forge-ui] {self.address_string()} {format % args}")

  def _json(self, payload: Any, status: int = 200):
    body = json.dumps(payload, default=str).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def _body(self) -> dict[str, Any]:
    length = int(self.headers.get("Content-Length", "0"))
    return json.loads(self.rfile.read(length) or b"{}")

  def _static(self, name: str):
    path = (self.static_root / name).resolve()
    if self.static_root.resolve() not in path.parents and path != self.static_root.resolve(): return self.send_error(404)
    if not path.is_file(): return self.send_error(404)
    body = path.read_bytes()
    self.send_response(200)
    self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def _session_route(self):
    match = re.fullmatch(r"/api/sessions/([a-f0-9]+)(?:/(.*))?", urlparse(self.path).path)
    if not match: return None, None
    return self.engine.session(match.group(1)), match.group(2) or ""

  def do_GET(self):
    path = urlparse(self.path).path
    try:
      if path == "/": return self._static("index.html")
      if path in {"/app.js", "/style.css", "/kernels.css"}: return self._static(path[1:])
      if path == "/api/health": return self._json({"ok": True, "backend": self.engine.backend.name,
        "capabilities": asdict(self.engine.backend.capabilities), "runtime_fingerprint": asdict(self.engine.runtime_fingerprint())})
      if path == "/api/sessions": return self._json(self.engine.sessions())
      if path == "/api/optimization": return self._json(self.engine.optimization_state())
      session, tail = self._session_route()
      if session is None: return self.send_error(404)
      if tail == "": return self._json(session.snapshot())
      if tail == "profile": return self._json(self.engine.profile(session.session_id))
      if tail == "trace": return self._json(session.trace.to_dict())
      if tail == "trace/chrome": return self._json(session.trace.chrome_trace())
      return self.send_error(404)
    except KeyError as exc: return self._json({"error": str(exc)}, 404)
    except Exception as exc: return self._json({"error": str(exc), "type": type(exc).__name__}, 500)

  def do_POST(self):
    path = urlparse(self.path).path
    try:
      body = self._body()
      if path == "/api/sessions": return self._json(self.engine.create_session().snapshot(), HTTPStatus.CREATED)
      if path == "/api/recipes/import":
        result = self.engine.execute_explicit_ui_tool("import_forge_recipe", {"path": str(body.get("path", ""))},
          str(body.get("reason", "Import portable optimization recipe from local UI")))
        return self._json(result, HTTPStatus.CREATED)
      if path == "/api/recipes/export":
        result = self.engine.execute_explicit_ui_tool("export_forge_recipe", {"spec_id": str(body.get("spec_id", "")),
          "candidate_id": body.get("candidate_id"), "output": str(body.get("output", ""))},
          str(body.get("reason", "Export portable execution-stage optimization recipe")))
        return self._json(result, HTTPStatus.CREATED)
      if path == "/api/hooks/activate":
        result = self.engine.execute_explicit_ui_tool("activate_optimization_hook", {
          "candidate_id": str(body.get("candidate_id", "")), "reason": str(body.get("reason", "Deploy validated stage-specific optimization")),
          "allow_unknown_runtime": bool(body.get("allow_unknown_runtime", False))},
          str(body.get("reason", "Deploy validated stage-specific optimization")))
        return self._json(result, HTTPStatus.CREATED)
      if path == "/api/hooks/deactivate":
        result = self.engine.execute_explicit_ui_tool("deactivate_optimization_hook", {
          "activation_id": str(body.get("activation_id", "")), "reason": str(body.get("reason", "Rollback local optimization hook"))},
          str(body.get("reason", "Rollback local optimization hook")))
        return self._json(result)
      session, tail = self._session_route()
      if session is None: return self.send_error(404)
      if tail == "messages":
        events = session.send(str(body.get("content", "")), int(body.get("max_tokens", 512)), float(body.get("temperature", 0.0)))
        return self._json({"events": [asdict(x) for x in events], "session": session.snapshot()})
      if tail == "tools/approve":
        reason = str(body.get("reason", "Approved from Radeon Forge UI"))
        token = self.engine.grant_for_pending_tool(session.session_id, reason)
        events = session.approve_tool(token, int(body.get("max_tokens", 512)))
        return self._json({"events": [asdict(x) for x in events], "session": session.snapshot()})
      if tail == "tools/reject":
        event = session.reject_tool(str(body.get("reason", "Rejected by user")))
        return self._json({"event": asdict(event), "session": session.snapshot()})
      return self.send_error(404)
    except (ValueError, RuntimeError, FileNotFoundError) as exc: return self._json({"error": str(exc), "type": type(exc).__name__}, 400)
    except Exception as exc: return self._json({"error": str(exc), "type": type(exc).__name__}, 500)


def build_backend(args):
  if args.backend == "scripted": return ScriptedBackend()
  if args.model is None: raise SystemExit("--model is required for --backend tinygrad-llama")
  command = [sys.executable, "-m", "extra.radeon_forge.backends.tinygrad_llama_worker", "--model", str(args.model),
             "--size", args.size, "--max-context", str(args.max_context)]
  if args.tokenizer: command += ["--tokenizer", str(args.tokenizer)]
  if args.quantize: command += ["--quantize", args.quantize]
  return JsonlProcessBackend(command)


def main():
  parser = argparse.ArgumentParser(description="Radeon Forge local inference and profiling UI")
  parser.add_argument("--workspace", type=Path, default=Path.cwd())
  parser.add_argument("--backend", choices=("scripted", "tinygrad-llama"), default="scripted")
  parser.add_argument("--model", type=Path)
  parser.add_argument("--tokenizer", type=Path)
  parser.add_argument("--size", choices=("1B", "8B", "70B", "405B"), default="1B")
  parser.add_argument("--quantize", choices=("int8", "nf4", "float16", "fp8"))
  parser.add_argument("--max-context", type=int, default=8192)
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=7790)
  args = parser.parse_args()
  if args.host not in {"127.0.0.1", "localhost", "::1"}: raise SystemExit("Forge UI binds to loopback only")
  engine = ForgeEngine(build_backend(args), args.workspace)
  ForgeRequestHandler.engine = engine
  server = ThreadingHTTPServer((args.host, args.port), ForgeRequestHandler)
  print(f"Radeon Forge UI: http://{args.host}:{args.port} backend={engine.backend.name} workspace={args.workspace.resolve()}")
  try: server.serve_forever()
  finally: engine.close()


if __name__ == "__main__": main()
