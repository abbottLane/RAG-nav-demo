# Make a venv with python3.8 and install requirements
venv: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || virtualenv .venv
	. .venv/bin/activate; pip install -Ur requirements.txt
	touch .venv/touchfile

test: venv
	. .venv/bin/activate; python pgvec.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

demo:
	. .venv/bin/activate; python demo.py

yaml: venv
	wget https://github.com/milvus-io/milvus/releases/download/v2.2.13/milvus-standalone-docker-compose.yml -O docker-compose.yml

start: yaml
	docker-compose up -d

stop:
	sudo docker-compose down

clean-volumes:
	sudo rm -rf volumes

client: venv
	. .venv/bin/activate; python app/gui/client.py