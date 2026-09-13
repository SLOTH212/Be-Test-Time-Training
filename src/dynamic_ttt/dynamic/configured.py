"""Architecture binding of the frozen recomputed search implementation."""
import ast, types
from pathlib import Path
from dynamic_ttt.eval.runtime import fixed_actions, action_layers

def runner(cfg):
 from dynamic_ttt.dynamic import search
 path=Path(search.__file__)
 tree=ast.parse(path.read_text())
 class Chunk(ast.NodeTransformer):
  def visit_Constant(self,n):
   return ast.copy_location(ast.Constant(cfg['ttt_chunk_size']),n) if type(n.value) is int and n.value==1024 else n
 ns={'__name__':'configured_dynamic_search','__file__':str(path)}
 exec(compile(ast.fix_missing_locations(Chunk().visit(tree)),str(path),'exec'),ns)
 layers=list(cfg['ttt_layers']);actions=fixed_actions(layers)
 ns.update(LAYERS=layers,ACTIONS=actions,ACTION_LAYERS=action_layers(layers),ACTION_MODE={a:a for a in actions})
 return types.SimpleNamespace(**ns)
