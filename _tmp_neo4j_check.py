from neo4j import GraphDatabase
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))
with driver.session(database="neo4j") as session:
    rec = session.run("RETURN 1 AS ok, currentUser() AS user, currentDatabase() AS db").single()
    print(dict(rec))
driver.close()