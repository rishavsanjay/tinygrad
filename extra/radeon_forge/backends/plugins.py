from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkerPlugin:
  name: str
  module: str
  description: str
  required: tuple[str, ...]
  optional: tuple[str, ...] = ()
  defaults: Mapping[str, Any] = field(default_factory=dict)

  def command(self, values: Mapping[str, Any]) -> list[str]:
    config={**dict(self.defaults),**{str(k):v for k,v in values.items() if v is not None}}
    missing=[name for name in self.required if config.get(name) in (None,"")]
    if missing: raise ValueError(f"worker plugin {self.name!r} is missing required fields: {missing}")
    allowed=set(self.required)|set(self.optional)
    unknown=sorted(set(config)-allowed)
    if unknown: raise ValueError(f"worker plugin {self.name!r} received unsupported fields: {unknown}")
    command=[sys.executable,"-m",self.module]
    for name in self.required+self.optional:
      if name not in config or config[name] is None: continue
      value=config[name]
      flag="--"+name.replace("_","-")
      if isinstance(value,bool):
        if value: command.append(flag)
      elif isinstance(value,(str,int,float,Path)): command += [flag,str(value)]
      elif isinstance(value,Sequence) and not isinstance(value,(str,bytes,bytearray)):
        for item in value: command += [flag,str(item)]
      else: raise TypeError(f"unsupported worker option {name}: {type(value).__name__}")
    return command


class WorkerPluginRegistry:
  def __init__(self): self._plugins: dict[str,WorkerPlugin]={}

  def register(self,plugin:WorkerPlugin) -> None:
    if not plugin.name or plugin.name in self._plugins: raise ValueError(f"duplicate or empty worker plugin {plugin.name!r}")
    if not plugin.module.startswith("extra.radeon_forge.backends."):
      raise ValueError("worker plugins must resolve to an in-tree local Radeon Forge backend module")
    self._plugins[plugin.name]=plugin

  def get(self,name:str) -> WorkerPlugin:
    try:return self._plugins[name]
    except KeyError as exc:raise KeyError(f"unknown local worker plugin {name!r}; available={sorted(self._plugins)}") from exc

  def list(self) -> list[dict[str,Any]]:
    return [{"name":item.name,"module":item.module,"description":item.description,
             "required":list(item.required),"optional":list(item.optional),"defaults":dict(item.defaults)}
            for item in sorted(self._plugins.values(),key=lambda x:x.name)]


def default_worker_plugins() -> WorkerPluginRegistry:
  registry=WorkerPluginRegistry()
  registry.register(WorkerPlugin(
    name="legacy-llama",
    module="extra.radeon_forge.backends.tinygrad_llama_worker",
    description="Resident tinygrad Llama-family worker with exact KV accounting, stage hooks and kernel profiling.",
    required=("model",),
    optional=("tokenizer","size","quantize","max_context","seed"),
    defaults={"size":"1B","max_context":8192,"seed":42},
  ))
  return registry
