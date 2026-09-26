"""Extract compute-transport integers without importing or loading Mesa bindings."""
import ast, pathlib, re

root=pathlib.Path(__file__).resolve().parents[2]
transport=(root/'tinygrad/runtime/ops_qcom.py').read_text()+(root/'tinygrad/runtime/support/qcom_profile.py').read_text()
prefixes={n.upper() for n in re.findall(r'qreg\.([a-zA-Z0-9_]+)',transport)}
direct=set(re.findall(r'mesa\.([a-zA-Z0-9_]+)',transport))
constants={}
for node in ast.walk(ast.parse((root/'tinygrad/runtime/autogen/mesa.py').read_text())):
  if isinstance(node,ast.NamedExpr): targets=[node.target]
  elif isinstance(node,ast.Assign): targets=node.targets
  else: continue
  if not isinstance(node.value,ast.Constant) or type(node.value.value) is not int: continue
  for target in targets:
    if isinstance(target,ast.Name) and (target.id in direct or any(target.id=='REG_'+p or target.id.startswith(p+'_') for p in prefixes)):
      constants[target.id]=node.value.value
header = ('# Reference notices: extra/qcom_gpu_driver/ADRENO_REFERENCE_LICENSE.\n'
          '# Adreno compute transport constants; no shared-library bindings.\n'
          '# Extracted from autogen/mesa.py (Mesa 26.2.1, da14d65e4499e66468094be52bff9ea0915a695e).\n'
          '# Regenerate: python extra/qcom_gpu_driver/gen_adreno_constants.py\n')
(root/'tinygrad/runtime/autogen/adreno.py').write_text(header+''.join(f'{n} = {v:#x}\n' for n,v in sorted(constants.items())))
