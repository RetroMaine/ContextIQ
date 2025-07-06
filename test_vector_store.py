# test_vector_store.py
import traceback
from implementation import MilvusVectorStore

def main():
    store = MilvusVectorStore(
        host="localhost",    # adjust if your Milvus is elsewhere
        port=19530,
        collection_name="default"
    )

    print("→ Instantiated MilvusVectorStore")

    try:
        store.connect()
        print("✅ connect() succeeded")
    except Exception as e:
        print("❌ connect() failed:")
        traceback.print_exc()

    # since we probably don't have real data yet,
    # we'll just test close()

    try:
        store.close()
        print("✅ close() succeeded")
    except Exception as e:
        print("❌ close() failed:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
