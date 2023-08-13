import json
import sentence_transformers
import logging

logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")


def load_data(filepath):
    """
    Loads data from a json file, cleans fields, and embeds text
    """
    data = []
    names = []
    categories = []
    descriptions = []
    with open(filepath, "r") as f:
        for line in f:
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


def embed(texts):
    """
    embeds text using the universal sentence encoder
    """
    logging.info(f"Embedding {len(texts)} contexts")
    embeddings = EMBEDDING_MODEL.encode(texts)
    return embeddings
