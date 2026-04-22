# Agent Skill: `/api/agent/chunk_search`

## Purpose

This endpoint is intended for external LLM agents and provides structured retrieval over `LIGHTRAG_DOC_CHUNKS` (`doc_chunks`).

Current scope:

- Supports structured search for `doc_chunks`
- Future agent-facing graph search APIs can be added under `/api/agent`
- The current implementation requires PostgreSQL-backed `text_chunks` storage

## Endpoint

- Method: `POST`
- Path: `/api/agent/chunk_search`

## Authentication

This endpoint follows the existing LightRAG API authentication rules:

- `X-API-Key`
- or OAuth2 token

## Workspace Routing

Workspace can be provided in any of these ways:

- Request body field `workspace`
- Query parameter `workspace`
- Header `LIGHTRAG-WORKSPACE`

Priority is the same as the rest of the API:

- `query > body > header > default workspace`

## Request Schema

```json
{
  "workspace": "default",
  "chunk_id": "chunk-xxx",
  "doc_id": "doc-xxx",
  "content": "exact full content match",
  "content_like": "substring search text",
  "content_type": ["text", "table"],
  "top_k": 20
}
```

Field rules:

- `chunk_id`: exact match on `LIGHTRAG_DOC_CHUNKS.id`
- `doc_id`: exact match on `LIGHTRAG_DOC_CHUNKS.full_doc_id`
- `content`: exact match on `content`
- `content_like`: case-insensitive substring match on `content`
- `content_type`: accepts a string or a string array; matching is case-insensitive
- `top_k`: maximum number of returned rows, default `20`, max `200`

Compatibility notes:

- `doc_id` also accepts the alias `full_doc_id`
- All provided filters are combined with `AND`
- `content` and `content_like` may be used together

## Response Schema

```json
{
  "chunks": [
    {
      "chunk_id": "chunk-xxx",
      "doc_id": "doc-xxx",
      "content": "chunk content",
      "content_type": "text"
    }
  ],
  "count": 1
}
```

Each returned chunk contains only:

- `chunk_id`
- `doc_id`
- `content`
- `content_type`

## Recommended Agent Usage

1. Use `chunk_id` when the exact chunk identifier is known.
2. Add `doc_id` when the search should stay inside a known document.
3. Use `content` for exact lookup.
4. Use `content_like` for substring retrieval.
5. Keep `content_type` explicit when possible, such as `text`, `table`, or `image`.
6. Always send a bounded `top_k`.

## Example Request

```bash
curl -X POST "http://localhost:9621/api/agent/chunk_search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "workspace": "default",
    "doc_id": "doc-123",
    "content_like": "LightRAG",
    "content_type": ["text"],
    "top_k": 10
  }'
```

## Example Response

```json
{
  "chunks": [
    {
      "chunk_id": "chunk-001",
      "doc_id": "doc-123",
      "content": "LightRAG is an advanced Retrieval-Augmented Generation framework.",
      "content_type": "text"
    }
  ],
  "count": 1
}
```

# Agent Skill: `/api/agent/entity_serach`

## Purpose

This endpoint is intended for external LLM agents and provides structured retrieval over graph entities.

Current storage support:

- PostgreSQL AGE graph storage
- Neo4j graph storage

## Endpoint

- Method: `POST`
- Path: `/api/agent/entity_serach`

Note:

- The path keeps the current project spelling: `entity_serach`

## Authentication

This endpoint follows the existing LightRAG API authentication rules:

- `X-API-Key`
- or OAuth2 token

## Workspace Routing

Workspace can be provided in any of these ways:

- Request body field `workspace`
- Query parameter `workspace`
- Header `LIGHTRAG-WORKSPACE`

Priority is:

- `query > body > header > default workspace`

## Request Schema

```json
{
  "workspace": "default",
  "description": "retrieval",
  "entity_name": "LightRAG",
  "entity_name_like": "light",
  "entity_type": ["project", "organization"],
  "entity_type_like": ["proj", "org"]
}
```

Field rules:

- `description`: case-insensitive substring match on `description`
- `entity_name`: exact match on graph node `entity_id`
- `entity_name_like`: case-insensitive substring match on graph node `entity_id`
- `entity_type`: exact match on `entity_type`; accepts a string or a string array
- `entity_type_like`: case-insensitive substring match on `entity_type`; accepts a string or a string array

Compatibility notes:

- `entity_name` also accepts the alias `entity_id`
- `description` also accepts the alias `description_like`
- All provided filters are combined with `AND`

## Response Schema

```json
{
  "entities": [
    {
      "description": "retrieval framework",
      "entity_id": "LightRAG",
      "entity_name": "LightRAG",
      "source_id": ["chunk-1", "chunk-2"],
      "entity_type": "project"
    }
  ],
  "count": 1
}
```

Each returned entity contains only:

- `description`
- `entity_id`
- `entity_name`
- `source_id`
- `entity_type`

`source_id` is always returned as a list split from the stored `<SEP>`-joined value.
`entity_id` is the canonical value that should be passed into `/api/agent/submit` as `entity_ids`.

## Recommended Agent Usage

1. Use returned `entity_id` as the submit-safe identifier for `/api/agent/submit`.
2. Use `entity_name_like` when only part of the identifier is known.
3. Use `description` to retrieve entities by semantic text fragments already present in the graph.
4. Use `entity_type` to constrain the result set to exact categories.
5. Use `entity_type_like` when type naming is inconsistent across data sources.

## Example Request

```bash
curl -X POST "http://localhost:9621/api/agent/entity_serach" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "workspace": "default",
    "description": "retrieval",
    "entity_name_like": "light",
    "entity_type": ["project"],
    "entity_type_like": ["proj"]
  }'
```

## Example Response

```json
{
  "entities": [
    {
      "description": "retrieval framework",
      "entity_id": "LightRAG",
      "entity_name": "LightRAG",
      "source_id": ["chunk-1", "chunk-2"],
      "entity_type": "project"
    }
  ],
  "count": 1
}
```

# Agent Skill: `/api/agent/submit`

## Purpose

This endpoint stores agent-submitted entity ids and chunk ids in a PostgreSQL-backed cache table.

The cache is designed for short-lived agent coordination data, not long-term business storage.

## Endpoint

- Method: `POST`
- Path: `/api/agent/submit`

## Authentication

This endpoint follows the existing LightRAG API authentication rules:

- `X-API-Key`
- or OAuth2 token

## Workspace Routing

Workspace can be provided in any of these ways:

- Request body field `workspace`
- Query parameter `workspace`
- Header `LIGHTRAG-WORKSPACE`

Priority is:

- `query > body > header > default workspace`

## Request Schema

```json
{
  "workspace": "default",
  "agent_submit_id": "submit-001",
  "entity_ids": ["entity-1", "entity-2"],
  "chunk_ids": ["chunk-1", "chunk-2"]
}
```

Field rules:

- `agent_submit_id`: required cache key for the submitted payload
- `entity_ids`: entity ids to store; accepts a string or a string array
- `chunk_ids`: chunk ids to store; accepts a string or a string array
- At least one of `entity_ids` or `chunk_ids` must be provided
- Every chunk id must start with `chunk-`

Compatibility notes:

- `entity_ids` also accepts the alias `entity_id`
- `chunk_ids` also accepts the alias `chunk_id`
- Re-submitting the same `agent_submit_id` in the same workspace overwrites the previous payload and refreshes TTL

## Response Schema

```json
{
  "status": "submitted",
  "workspace": "default",
  "agent_submit_id": "submit-001",
  "entity_ids": ["entity-1", "entity-2"],
  "chunk_ids": ["chunk-1", "chunk-2"],
  "expires_at": "2026-04-15T12:00:00Z"
}
```

## Cache Retention And Cleanup

Retention policy:

- Each entry expires 24 hours after the latest write

Cleanup design:

- The cache uses a dedicated PostgreSQL table: `LIGHTRAG_AGENT_SUBMIT_CACHE`
- The table stores an `expires_at` column
- The API performs indexed, opportunistic cleanup before submit writes
- Cleanup runs on a time window instead of every request, reducing repeated delete pressure
- Expired rows are eligible for removal once `expires_at <= CURRENT_TIMESTAMP`

This approach keeps cleanup simple, deterministic, and independent of external schedulers.

## Recommended Agent Usage

1. Use a stable `agent_submit_id` when the same logical submission may be retried.
2. Send only the ids needed for downstream coordination.
3. Treat this endpoint as ephemeral cache storage, not an audit log.
4. Re-submit the same `agent_submit_id` if the payload must be refreshed for another 24 hours.

## Example Request

```bash
curl -X POST "http://localhost:9621/api/agent/submit" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "workspace": "default",
    "agent_submit_id": "submit-001",
    "entity_ids": ["entity-1"],
    "chunk_ids": ["chunk-9"]
  }'
```

## Example Response

```json
{
  "status": "submitted",
  "workspace": "default",
  "agent_submit_id": "submit-001",
  "entity_ids": ["entity-1"],
  "chunk_ids": ["chunk-9"],
  "expires_at": "2026-04-15T12:00:00Z"
}
```
