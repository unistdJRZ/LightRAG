from pathlib import Path
from lightrag.workspace_config import load_workspace_config
cfg = load_workspace_config(Path('workspace.yaml'))
print('ok')
print([w.id for w in cfg['workspaces']])