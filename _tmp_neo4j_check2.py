from neo4j import GraphDatabase
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "12345678"))
with driver.session(database="neo4j") as session:
    rec = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions[0] AS version LIMIT 1").single()
    print(dict(rec))
driver.close()