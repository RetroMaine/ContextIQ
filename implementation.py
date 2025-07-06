from interface import VectorStoreInterface

class MilvusVectorStore(VectorStoreInterface):
    def __init__(self, host: str = "localhost", port: int = 19530, collection_name: str = "default"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None

    def connect(self, **kwargs) -> None:
        from pymilvus import connections, Collection # type: ignore[import]
        connections.connect(alias="default", host=self.host, port=self.port)
        self.client = Collection(self.collection_name)
    
    def search(self, query_embedding, top_k, filter=None):
        results = self.client.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            expr=None  # adapt for filter expressions
        )
        return [(r.id, r.distance, r.entity.get("metadatas", {})) for hit in results for r in hit]
    
    def delete(self, ids):
        self.client.delete(expr=f"id in {ids}")

    def close(self):
        from pymilvus import connections # type: ignore[import]
        connections.disconnect(alias="default")
        