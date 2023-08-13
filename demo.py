# dc_pois.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Milvus
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
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
from app.utils import load_data
from app.utils import embed

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8


#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("dc_pois")
print(f"Does collection dc_pois exist in Milvus: {has}")

if has:
    ####
    ## 0. drop collection if exists...for now...
    print(fmt.format("Drop collection `dc_pois`"))
    utility.drop_collection("dc_pois")

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "pk"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|  "random"  |    Double  |                  |      "a double field"        |
# +-+------------+------------+------------------+------------------------------+
# |3|"embeddings"| FloatVector|     dim=8        |  "float vector with dim 8"   |
# +-+------------+------------+------------------+------------------------------+

"""
Create the following schema for the collection `dc_pois`:
            {
        "type": "Feature",
        "geometry": { "coordinates": [-77.065721, 38.911355], "type": "Point" },
        "properties": {
            "mapbox:id": "dba5f7f85f21e296cafae99871c38055762b59f745890066509acaa11c43963a",
            "foursquare": {
            "id": "c4fa616f2cb940b93aef20e7",
            "hours": {
                "friday": [["8:00", "16:00"]],
                "thursday": [["8:00", "16:00"]],
                "wednesday": [["8:00", "16:00"]],
                "tuesday": [["8:00", "16:00"]],
                "monday": [["8:00", "16:00"]]
            },
            "hours_popular": null,
            "hours_display": "Mon-Fri 8:00 AM-4:00 PM",
            "tel": "(202) 829-9233",
            "website": "https://www.coloradosecurity.com/contact-us",
            "email": "hfranklin@coloradosecurityagency.com",
            "rating": null,
            "provenance_rating": "4",
            "description": "",
            "price": "",
            "tips": null,
            "tastes": null,
            "photos": null,
            "popularity": "0.19",
            "clean": "",
            "crowded": "",
            "drivethrough": "",
            "servicequality": "",
            "noisy": "",
            "valueformoney": ""
            },
            "iso_3166_1": "US",
            "iso_3166_2": "US-DC",
            "mapbox:search:categories": "office",
            "addr:state": "District of Columbia",
            "addr:postcode": "20007",
            "name": "Colorado Security Agency Amanda",
            "addr:neighborhood": "Georgetown",
            "addr:full": "1622 Wisconsin Ave NW",
            "addr:country": "USA",
            "addr:city": "Washington",
            "addr:housenumber": "1622",
            "addr:street": "wisconsin ave nw"
        }
        }
"""
fields = [
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
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]

schema = CollectionSchema(fields, "Washington DC POIs")

print(fmt.format("Create collection `dc_pois`"))
dc_pois = Collection("dc_pois", schema, consistency_level="Strong")

################################################################################
# 3. insert data
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))
entities = load_data(filepath="data/us_dc_georgetown_with_details.json")

insert_result = dc_pois.insert(pd.DataFrame(entities))

dc_pois.flush()
print(f"Number of entities in Milvus: {dc_pois.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for dc_pois collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128},
}

dc_pois.create_index("embedding", index)

################################################################################
# 5. search, query, and hybrid search
# After data were inserted into Milvus and indexed, you can perform:
# - search based on vector similarity
# - query based on scalar filtering(boolean, int, etc.)
# - hybrid search based on vector similarity and scalar filtering.
#

# Before conducting a search or a query, you need to load the data in `dc_pois` into memory.
print(fmt.format("Start loading"))
dc_pois.load()

# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
user_queries = [
    "a security company that is open on weekends",
    "a mexican restaurant with taquitos",
    "a coffee shop that sells sandwiches",
    "a popular bar that has nightlife and a dinner menu",
]
vectors_to_search = [embed(x) for x in user_queries]

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = dc_pois.search(
    vectors_to_search,
    "embedding",
    search_params,
    limit=3,
    output_fields=["name", "category"],
)
end_time = time.time()

for hits in result:
    for hit in hits:
        print(
            f"hit: {hit}, name field: {hit.entity.get('name')}, category field: {hit.entity.get('category')}"
        )
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# query based on str filtering
print(fmt.format("Start querying with `name = Colorado Security Agency Amanda`"))

start_time = time.time()
result = dc_pois.query(
    expr="name == 'Colorado Security Agency Amanda'",
    output_fields=["name", "addr_full", "addr_street"],
)
end_time = time.time()

print(f"query result:\n-{result[0]}")
print(search_latency_fmt.format(end_time - start_time))

# -----------------------------------------------------------------------------
# pagination
# r1 = dc_pois.query(
#     expr="name == 'Colorado Security Agency Amanda", limit=4, output_fields=["name"]
# )
# r2 = dc_pois.query(
#     expr="name == 'Colorado Security Agency Amanda",
#     offset=1,
#     limit=3,
#     output_fields=["name"],
# )
# print(f"query pagination(limit=4):\n\t{r1}")
# print(f"query pagination(offset=1, limit=3):\n\t{r2}")


# -----------------------------------------------------------------------------
# hybrid search
print(
    fmt.format("Start hybrid searching with `name == Colorado Security Agency Amanda`")
)

start_time = time.time()
result = dc_pois.search(
    [vectors_to_search[0]],
    "embedding",
    search_params,
    limit=3,
    expr="name == 'Colorado Security Agency Amanda'",
    output_fields=["name", "category"],
)
end_time = time.time()

for hits in result:
    for hit in hits:
        print(f"hit: {hit}, name field: {hit.entity.get('name')}")
print(search_latency_fmt.format(end_time - start_time))

###############################################################################
# 6. delete entities by PK
# You can delete entities by their PK values using boolean expressions.
# ids = insert_result.primary_keys

# expr = f'pk in ["{ids[0]}" , "{ids[1]}"]'
# print(fmt.format(f"Start deleting with expr `{expr}`"))

# result = dc_pois.query(expr=expr, output_fields=["name", "embedding"])
# print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

# dc_pois.delete(expr)

# result = dc_pois.query(expr=expr, output_fields=["name", "embedding"])
# print(f"query after delete by expr=`{expr}` -> result: {result}\n")


###############################################################################
# 7. drop collection
# Finally, drop the dc_pois collection
print(fmt.format("Drop collection `dc_pois`"))
utility.drop_collection("dc_pois")
