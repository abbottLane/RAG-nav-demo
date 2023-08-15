import time

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import pandas as pd
from utils import load_data
from utils import embed

import logging

logging.basicConfig(level=logging.INFO)

logging.info("Connecting to Milvus deployment...")
connections.connect("default", host="localhost", port="19530")

# DEMO ONLY: Drop collections if they exist. We want to start fresh.
if utility.has_collection("dc_pois"):
    logging.info("Deleting collection dc_pois...")
    utility.drop_collection("dc_pois")

if utility.has_collection("duvall_pois"):
    logging.info("Deleting collection duvall_pois...")
    utility.drop_collection("duvall_pois")


def index_collection(fields, name, filepath, description):
    schema = CollectionSchema(fields, description)

    logging.info(f"Creating collection {name}...")
    collection = Collection(name, schema, consistency_level="Strong")

    logging.info(f"Loading pois data from {filepath}...")
    entities = load_data(filepath=filepath)

    insert_result = collection.insert(pd.DataFrame(entities))

    collection.flush()
    logging.info(f"Number of entities in {name} index: {collection.num_entities}")
    return collection


def benchmark_index_collection(fields, name, filepath, description, iterations=5):
    schema = CollectionSchema(fields, description)

    for i in range(iterations):
        logging.info(f"BENCHMARKING INDEXING: Creating collection {name}...")
        collection = Collection(name, schema, consistency_level="Strong")
        logging.info(f"Loading pois data from {filepath}...")

        start_embedding = time.time()
        entities = load_data(filepath=filepath, duplication=i + 1)
        end_embedding = time.time()
        logging.info(f"Embedding time: {end_embedding - start_embedding}")

        start_indexing_time = time.time()
        insert_result = collection.insert(pd.DataFrame(entities))
        end_indexing_time = time.time()
        logging.info(f"Indexing time: {end_indexing_time - start_indexing_time}")

        collection.flush()

        # delete the collection so we can start fresh
        logging.info(f"Deleting collection {name}...")
        utility.drop_collection(name)


VECTOR_DB_SCHEMA = [
    FieldSchema(
        name="mbx_id",
        dtype=DataType.VARCHAR,
        max_length=200,
        is_primary=True,
        auto_id=False,
    ),
    FieldSchema(name="latitude", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="longitude", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="addr_full", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="addr_street", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]

# benchmark_index_collection(
#     VECTOR_DB_SCHEMA,
#     "dc_pois",
#     "data/us_dc_georgetown_with_details.json",
#     description="Georgetown, Washington DC POIs",
#     iterations=20,
# )  # comment this out when you actually want to use the UI, otherwise you'll run bechmarking every time you start the app

dc_pois = index_collection(
    VECTOR_DB_SCHEMA,
    "dc_pois",
    "data/us_dc_georgetown_with_details.json",
    description="Georgetown, Washington DC POIs",
)
duvall_pois = index_collection(
    VECTOR_DB_SCHEMA,
    "duvall_pois",
    "data/us_duvall_wa_with_details.json",
    description="Duvall, Washington POIs",
)


logging.info("Creating index: IVF_FLAT and L2...")
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}
dc_pois.create_index("embedding", index)
duvall_pois.create_index("embedding", index)


class VectorDB:
    def __init__(self) -> None:
        self.collections = {
            "Georgetown, DC": dc_pois,
            "Duvall, WA": duvall_pois,
        }

    def set_idx_by_location(self, location):
        logging.info(f"Setting index to {location}")
        if location == "Georgetown, DC":
            self.idx = self.collections["Georgetown, DC"].load()
        elif location == "Duvall, WA":
            self.idx = self.collections["Duvall, WA"].load()

    def search(self, collection, query, top_k=5):
        """
        Searches the vector database for the top k results
        """
        logging.info(f"Searching {collection} for {query}...")
        start = time.time()
        query_embedding = embed([query])
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        results = self.collections[collection].search(
            query_embedding,
            "embedding",
            search_params,
            limit=top_k,
            output_fields=["name", "category"],
        )
        end = time.time()
        logging.info(
            f"\tEmbedding query + vectorDB lookup time: {(end - start)} seconds"
        )
        return results
