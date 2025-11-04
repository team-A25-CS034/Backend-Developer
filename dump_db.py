import asyncio
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import json
from typing import Any

load_dotenv()

MONGO_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('MONGODB_DATABASE', 'machine_monitoring_db')
COLLECTION_NAME = os.getenv('MONGODB_COLLECTION', 'machine_monitoring')


def to_serializable(obj: Any) -> Any:
    """Convert common BSON types to JSON-serializable values."""
    # Avoid importing bson directly to keep dependencies small.
    try:
        from bson.objectid import ObjectId
        from datetime import datetime
    except Exception:
        ObjectId = None
        datetime = None

    # ObjectId
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    # datetime
    if datetime is not None and isinstance(obj, datetime):
        return obj.isoformat()

    # lists and dicts
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    # fallback: return as-is
    return obj


async def main(limit: int = 1000):
    if not MONGO_URI:
        print('MONGODB_URI not set in environment (.env file missing or MONGODB_URI unset)')
        return

    client = AsyncIOMotorClient(MONGO_URI)
    try:
        db = client[DB_NAME]
        print(f"Connected to DB: {DB_NAME}")
        collections = await db.list_collection_names()
        print('Collections:', collections)

        if COLLECTION_NAME not in collections:
            print(f"Collection '{COLLECTION_NAME}' not found in database. Available collections: {collections}")
            return

        coll = db[COLLECTION_NAME]
        count = await coll.count_documents({})
        print(f"Documents in '{COLLECTION_NAME}': {count}")

        cursor = coll.find({}).sort('timestamp', 1).limit(limit)
        docs = await cursor.to_list(length=limit)

        print('\nSample documents (up to {limit}):')
        for i, d in enumerate(docs, start=1):
            serial = to_serializable(d)
            print(json.dumps(serial, indent=2, ensure_ascii=False))
            if i >= 5:
                break

    except Exception as e:
        print('Error while reading DB:', e)
    finally:
        client.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Dump documents from MongoDB collection for inspection')
    parser.add_argument('--limit', type=int, default=1000, help='Maximum number of documents to fetch')
    args = parser.parse_args()

    asyncio.run(main(limit=args.limit))
