from pathlib import Path
from lightrag.tools.migrate_workspaces_to_postgres import read_embedding_dim
for ws in ['msre','msr_1','ornl_test','batch_test']:
    p = Path('rag_storage')/ws/'vdb_entities.json'
    print(ws, read_embedding_dim(p))