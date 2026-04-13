import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        host="localhost",
        port=5455,
        user="postgresUser",
        password="postgresPW",
        database="postgresDB",
    )
    row = await conn.fetchrow("select current_database() as db, current_user as usr")
    print("connected")
    print(dict(row))
    await conn.close()

asyncio.run(main())