# Learn2RAG basic implementation

## Requirements

- Python 3.13.5
- poetry
- direnv
- Docker
- Ollama

To install the dependencies run:
```
poetry install
poetry shell
```

Configure options in ``.envrc``:
```
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_AUTH=""
```
You might need to run ``direnv allow`` in the same directory.

### Ollama (not tested by USU!)
```
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -xzvf ollama-linux-amd64.tgz
bin/ollama serve
```
```
curl http://localhost:11434/api/pull -d '{"model": "llama3.3:70b"}'
```

### Start the containers for qdrant and ??? by running
```
docker compose up -d
```

### Optional: get repliqa dataset for recall evaluation

```
python data/get_repliqa_pdfs.py
```

### Optional: get datasets
```
tests/data/wikibooks/pages-articles.xml.bz2:
	mkdir --parents $$(dirname $@)
	wget --output-document $@ "https://files.dice-research.org/datasets/Wikibooks/20250501/dewikibooks-20250501-pages-articles.xml.bz2"

tests/data/html/AIAct.html:
	mkdir --parents $$(dirname $@)
	wget --output-document $@ "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=OJ:L_202401689"
```

### Explanations

``user_config.json`` are the configurations we expect to get from the UI
``opt_config.json`` are values we later want to optimize
``loaded_documents.json`` should mock the input extracted from AP2 importers


