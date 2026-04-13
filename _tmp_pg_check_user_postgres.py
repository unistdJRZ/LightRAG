import asyncio
import asyncpg

async def main():
    conn = await asyncpg.connect(
        host="localhost",
        port=5455,
        user="lightrag",
        password="123",
        database="postgres",
    )
    row = await conn.fetchrow("select current_database() as db, current_user as usr")
    print("connected")
    print(dict(row))
    await conn.close()

asyncio.run(main())