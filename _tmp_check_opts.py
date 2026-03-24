import asyncio
from lightrag.api.config import parse_args
from lightrag.llm.binding_options import OllamaEmbeddingOptions
import ollama

args = parse_args()
opts = OllamaEmbeddingOptions.options_dict(args)
model = args.embedding_model or 'bge-m3:latest'
host = args.embedding_binding_host

async def main():
    client = ollama.AsyncClient(host=host, timeout=30)
    resp = await client.embed(model=model, input=['????'], options=opts)
    print('ok', host, model, len(resp['embeddings'][0]))

asyncio.run(main())
