import ast
from pathlib import Path
import sys
import types

import numpy as np

from test_fusion import SOURCE

HERE = Path(__file__).resolve().parent
OUT = HERE / 'smoke_plots'


class SmallRun(ast.NodeTransformer):
    replacements = {
        'BASE_CLUSTER_COUNTS': '(4, 8)',
        'BASE_MIN_CLUSTER_SIZE': '2',
        'MIN_CLUSTER_SIZE': '2',
        'MAX_CLUSTERS': '24',
        'STD_LIMITS': 'np.array([.5, .5, 60.])',
        'N_PLOT': '30',
        'FUSION_CACHE_FILE': repr(str(HERE / 'smoke_cache.npz')),
        'output_directory': 'Path(' + repr(str(OUT)) + ')',
    }

    def visit_Assign(self, node):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for name in names:
            if name in self.replacements:
                node.value = ast.parse(self.replacements[name], mode='eval').body
            if name == 'source_ids' and isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == 'valid_ids':
                node.value = ast.parse('valid_ids[:24]', mode='eval').body
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'Pool':
            node.keywords = [ast.keyword(arg='processes', value=ast.Constant(2))]
        return self.generic_visit(node)


tree = SmallRun().visit(ast.parse(SOURCE.read_text()))
ast.fix_missing_locations(tree)
module = types.ModuleType('ga16_smoke')
module.__file__ = str(SOURCE)
sys.modules[module.__name__] = module
exec(compile(tree, str(SOURCE), 'exec'), module.__dict__)
assert len(module.labels) > 0
assert np.any(module.labels >= 0)
assert len(list(OUT.glob('*.png'))) >= 8
print('SMOKE PASS: actual integrations, fusion, representatives and both plotting frames', flush=True)
