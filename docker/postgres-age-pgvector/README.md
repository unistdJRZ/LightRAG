# PostgreSQL + Apache AGE + pgvector

This image extends `apache/age` by compiling and installing `pgvector`, then enables both extensions on first database initialization.

## Files

- `Dockerfile`: builds a combined PostgreSQL image with AGE and pgvector
- `initdb/00-enable-extensions.sh`: enables `vector` and `age` for the initial database
- `docker-compose.windows.yml`: Windows-friendly example with the bind mount fixed to `G:/postgredocker_data`

## Build

```powershell
docker compose -f docker/postgres-age-pgvector/docker-compose.windows.yml build
```

## Start

Create the bind-mount directory first:

```powershell
New-Item -ItemType Directory -Force G:\postgredocker_data
```

For PostgreSQL 18 based images, the bind mount must target `/var/lib/postgresql`, not `/var/lib/postgresql/data`.

Then start the container:

```powershell
docker compose -f docker/postgres-age-pgvector/docker-compose.windows.yml up -d
```

## Verify

Check the container:

```powershell
docker ps
docker exec -it lightrag-postgres-age psql -U postgresUser -d postgresDB
```

Inside `psql`, verify the extensions:

```sql
\dx
```

You should see both `age` and `vector`.

## LightRAG Connection Settings

Use these values in `.env`:

```env
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage

POSTGRES_HOST=localhost
POSTGRES_PORT=5455
POSTGRES_USER=postgresUser
POSTGRES_PASSWORD=postgresPW
POSTGRES_DATABASE=postgresDB
```

`PGGraphStorage` now loads `age` on each connection before setting the AGE search path, so the container does not need extra session bootstrap commands.
