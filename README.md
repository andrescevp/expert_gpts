# Experts GPTs

This is a framework to configure LangChain GPTs for different tasks driven by a simple config file.

## Installation

1. Create you experts gpt config file. See [example.mygpt.yaml](example.mygpt.yaml) for an example.

```bash
cp example.mygpt.yaml mygpt.yaml
```

2. Install dependencies

```bash
# optionally create a virtual environment
python -m venv venv
./venv/bin/activate

# if you have not the postgres binaries installed on your system
# install it with your package manager
pip install psycopg-binary

pip install -r requirements.txt
```

3. Run the UI

```bash
docker-compose up -d
python -m ui.app
```
