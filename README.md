# Retrieval Augmented Generation for POI search

## Setup
These instructions are for Ubuntu. Mac may have a similar process. If you are on Windows I can't help you. 

### Prereqs
1. You must have docker and docker-compose installed on your machine
2. You must have virtualenv for python installed (`pip install virtualenv`)
3. Get the `us_dc_georgetown_with_details.json` dataset from wlane, and put it in the `RAG-nav-demo/data` folder
4. You must have tkinter installed on your system (for ubuntu: `sudo apt install python3-tk`)
5. (WIP, so optional for now) You need llama2 weights folder in the `RAG-nav-demo/models` directory, and the llama python bindings installed on your computer

### Running the app

1. Install python prereqs and pull the Milvus vector DB containers and start them with docker compose:
```
make start
```
2. Verify that the python env and Milvus are all set up correctly. If the following script runs to completion, you are good:
```
make demo
```
2. run the client:
```
make client
```
