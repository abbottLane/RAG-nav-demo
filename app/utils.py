import json
import sentence_transformers
import logging

logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")


def _load_dc_poi_format(filepath, duplication=1):
    data = []
    names = []
    categories = []
    descriptions = []
    with open(filepath, "r") as f:
        for line in f:
            for i in range(duplication):
                json_line = json.loads(line)
                if "addr:full" not in json_line["properties"]:
                    json_line["properties"]["addr:full"] = ""
                if "addr:street" not in json_line["properties"]:
                    json_line["properties"]["addr:street"] = ""
                name = json_line["properties"]["name"]
                category = " and ".join(
                    json_line["properties"]["mapbox:search:categories"].split(";")
                )
                description = ""
                if "description" in json_line["properties"]:
                    description = json_line["properties"]["tripadvisor"]["description"]
                descriptions.append(description)
                names.append(name)
                categories.append(category)
                data.append(
                    {
                        "mbx_id": str(json_line["properties"]["mapbox:id"]),
                        "latitude": str(json_line["geometry"]["coordinates"][0]),
                        "longitude": str(json_line["geometry"]["coordinates"][1]),
                        "name": str(name),
                        "addr_full": str(json_line["properties"]["addr:full"]),
                        "addr_street": str(json_line["properties"]["addr:street"]),
                        "category": str(category),
                        "description": str(description),
                    }
                )
    contexts = []
    for name_text, category_text in zip(names, categories):
        encoding_context = f"The place name is {name_text} and it is of type {category_text} and is described as {description}"
        contexts.append(encoding_context)
    embeddings = embed(contexts)
    for i, embedding in enumerate(embeddings):
        data[i]["embedding"] = embedding

    return data


def _load_simple_poi_format(filepath):
    data = []
    names = []
    categories = []
    descriptions = []
    with open(filepath, "r") as fp:
        for line in fp:
            json_line = json.loads(line)
            description = json_line["description"]
            name = json_line["name"]
            category = json_line["category"]
            descriptions.append(description)
            names.append(name)
            categories.append(category)
            data.append(
                {
                    "mbx_id": str(json_line["mbx_id"]),
                    "latitude": "N/A",
                    "longitude": "N/A",
                    "name": str(name),
                    "addr_full": "N/A",
                    "addr_street": "N/A",
                    "category": str(category),
                    "description": str(description),
                }
            )
    contexts = []
    for name_text, category_text in zip(names, categories):
        encoding_context = f"The place name is {name_text} and it is of type {category_text} and is described as {description}"
        contexts.append(encoding_context)
    embeddings = embed(contexts)
    for i, embedding in enumerate(embeddings):
        data[i]["embedding"] = embedding

    return data


def load_data(filepath, duplication=1):
    """
    Loads data from a json file, cleans fields, and embeds text.

    The DC dataset has a particular format given that it was sampled from Mapbox's POI dataset.
    The Duvall dataset was hand curated and has a simpler format.
    """
    if "us_dc_georgetown_with_details.json" in filepath:
        return _load_dc_poi_format(filepath, duplication=duplication)
    else:
        return _load_simple_poi_format(filepath)


def embed(texts):
    """
    embeds text using the universal sentence encoder
    """
    logging.info(f"Embedding {len(texts)} contexts")
    embeddings = EMBEDDING_MODEL.encode(texts)
    return embeddings


def parse_results(result):
    """
    Take the results from a vectorDB query, and turn it into a context string for the LLM query
    """
    results = []
    logging.info(f"Found {len(result)} results")
    for hits in result:
        logging.info(f"Found {len(hits)} hits")
        for hit in hits:
            logging.info(f"Found {hit}")
            results.append(f"place name: {hit.entity.get('name')}")
    return results
