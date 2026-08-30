from __future__ import annotations

import csv, json, math
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any,Iterable,Mapping


@dataclass(frozen=True)
class ParsedArtifact:
  path:str
  format:str
  parsed:bool
  event_count:int
  duration_us:float
  categories:Mapping[str,int]
  names:Mapping[str,int]
  wave_count:int|None=None
  instruction_classes:Mapping[str,int]|None=None
  reason:str=""


class UnsupportedSQTTFormat(ValueError):pass


def _manifest_path(value:str|Path)->Path:
  path=Path(value)
  if path.is_dir():path=path/"forge_capture_manifest.json"
  if not path.is_file():raise FileNotFoundError(path)
  return path


def _artifact_paths(manifest:Mapping[str,Any],root:Path)->list[Path]:
  paths=[]
  for item in manifest.get("artifacts",()):
    if not isinstance(item,Mapping):continue
    path=Path(str(item.get("path","")))
    if not path.is_absolute():path=root/path
    if path.is_file():paths.append(path)
  return paths


def _perfetto(path:Path)->ParsedArtifact:
  payload=json.loads(path.read_text(encoding="utf-8",errors="replace"))
  events=payload.get("traceEvents") if isinstance(payload,Mapping) else None
  if not isinstance(events,list):raise UnsupportedSQTTFormat("JSON does not contain traceEvents")
  categories=Counter();names=Counter();duration=0.0;wave_ids=set();instructions=Counter()
  for event in events:
    if not isinstance(event,Mapping):continue
    category=str(event.get("cat","uncategorized"));name=str(event.get("name","unnamed"))
    categories[category]+=1;names[name]+=1
    try:
      value=float(event.get("dur",0.0))
      if math.isfinite(value) and value>=0:duration+=value
    except (TypeError,ValueError):pass
    args=event.get("args",{})
    if isinstance(args,Mapping):
      for key in ("wave","wave_id","waveId"):
        if key in args:wave_ids.add(str(args[key]))
      for key in ("instruction_class","instruction_type","inst_class","opcode_class"):
        if key in args:instructions[str(args[key])]+=1
  return ParsedArtifact(str(path),"perfetto-json",True,len(events),duration,dict(categories),dict(names),
                        len(wave_ids) if wave_ids else None,dict(instructions) if instructions else None)


def _column(fieldnames:Iterable[str],candidates:tuple[str,...])->str|None:
  names=list(fieldnames)
  lowered={name.lower():name for name in names}
  for candidate in candidates:
    if candidate.lower() in lowered:return lowered[candidate.lower()]
  return None


def _table(path:Path)->ParsedArtifact:
  delimiter="\t" if path.suffix.lower() in {".tsv",".tab"} else ","
  with path.open(newline="",encoding="utf-8",errors="replace") as handle:
    reader=csv.DictReader(handle,delimiter=delimiter);fields=reader.fieldnames or []
    if not fields:raise UnsupportedSQTTFormat("table has no header")
    wave_col=_column(fields,("wave","wave_id","waveid"))
    inst_col=_column(fields,("instruction_class","instruction_type","inst_class","opcode_class","instruction"))
    name_col=_column(fields,("kernel_name","kernel","name","event"))
    category_col=_column(fields,("category","cat","type","event_type"))
    duration_col=_column(fields,("duration_us","duration","dur","elapsed_us"))
    start_col=_column(fields,("start_us","start","begin"));end_col=_column(fields,("end_us","end","finish"))
    rows=0;duration=0.0;waves=set();instructions=Counter();names=Counter();categories=Counter()
    for row in reader:
      rows+=1
      if wave_col and row.get(wave_col):waves.add(str(row[wave_col]))
      if inst_col and row.get(inst_col):instructions[str(row[inst_col])]+=1
      if name_col and row.get(name_col):names[str(row[name_col])]+=1
      if category_col and row.get(category_col):categories[str(row[category_col])]+=1
      try:
        if duration_col and row.get(duration_col) not in (None,""):value=float(row[duration_col])
        elif start_col and end_col:value=float(row[end_col])-float(row[start_col])
        else:value=0.0
        if math.isfinite(value) and value>=0:duration+=value
      except (TypeError,ValueError):pass
  recognized=bool(wave_col or inst_col or duration_col or (start_col and end_col))
  if not recognized:raise UnsupportedSQTTFormat("table lacks wave, instruction or timeline columns")
  return ParsedArtifact(str(path),"sqtt-table",True,rows,duration,dict(categories),dict(names),
                        len(waves) if waves else None,dict(instructions) if instructions else None)


def normalize_capture(value:str|Path)->dict[str,Any]:
  manifest_path=_manifest_path(value);root=manifest_path.parent
  manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
  parsed=[];unsupported=[]
  for path in _artifact_paths(manifest,root):
    if path==manifest_path:continue
    try:
      if path.suffix.lower()==".json":item=_perfetto(path)
      elif path.suffix.lower() in {".csv",".tsv",".tab"}:item=_table(path)
      else:raise UnsupportedSQTTFormat(f"no registered decoder for {path.suffix or 'binary'}")
      parsed.append(item)
    except (UnsupportedSQTTFormat,json.JSONDecodeError,csv.Error) as exc:
      unsupported.append(ParsedArtifact(str(path),"unknown",False,0,0.0,{}, {},reason=str(exc)))
  total_events=sum(item.event_count for item in parsed)
  total_duration=sum(item.duration_us for item in parsed)
  waves=sum(item.wave_count or 0 for item in parsed)
  instructions=Counter()
  for item in parsed:instructions.update(item.instruction_classes or {})
  return {"capture_id":manifest.get("capture_id"),"kind":manifest.get("kind"),"parsed":bool(parsed),
    "recognized_artifacts":[asdict(item) for item in parsed],"unsupported_artifacts":[asdict(item) for item in unsupported],
    "summary":{"recognized_count":len(parsed),"unsupported_count":len(unsupported),"event_count":total_events,
      "duration_us":total_duration,"wave_count":waves or None,"instruction_classes":dict(instructions)},
    "contract":"Only recognized structured exports are normalized. Raw ATT/SQTT binaries remain immutable evidence until a matching decoder is installed."}
