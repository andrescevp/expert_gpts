# Experts GPTs

This is a framework to configure LangChain GPTs for different tasks driven by a simple config file.

# DISCLAIMER: This repo is WIP and can change drastically in the future.

## Looking collaborators to improve it!

## Installation

1. Create configs files and edit them.

```bash
cp configs/mygpt.yaml.dist configs/mygpt.yaml # edit it
cp .env.dist .env # edit it specially OPENAI_API_KEY
```

2. Install dependencies

```bash
# optionally create a virtual environment
python -m venv venv
./venv/bin/activate

pip install -r requirements.txt
```

3. Run the UI

```bash
docker-compose up -d
python -m bin.load_docs configs/mygpt.yaml # load embeddings for a specific config
python -m ui.app
```

# How it works

## Experts

An expert is a GPT that is configured to answer a specific question. It is configured by a config file.
Each expert have a memory (Embeddings) that is available only for the expert under a llama index implementation.
This memory is query when the user ask a question to the expert.
The question is search first in the vector database and the result is passed to the GPT as a context.

## Chain

A chain is a sequence of experts. The output of an expert is the input of the next expert.
A chain have available all the experts as tools and other generic tools you can setup in the config file.
A chain have also a memory (Embeddings) that is available only for the chain under a llama index implementation.

## Embeddings VS History

Embeddings is a vector database that is used to store the context of the question and the answer of the expert.

History is a database that store the question and the answer of the expert or the chain per user with a session id.
It is implemented with MariaDB because it allows to use an implementation of fuzzy search using levenstein.
So what the history is is a selection of messages based in the proximity of the question.

@see: https://lucidar.me/en/web-dev/levenshtein-distance-in-mysql/

## Infra.

This project uses:

- RedisStack to store vector databases.
- LLama Index to manage the vector databases.
- MariaDB to store the history.
- Dash from Plotly to create the UI.

# Extending

If you want to add your self tools to the chain, you can do it following this steps:

1. Create a new file in the folder `my_code` with the name of your new fancy package. In our case we will call it `my_tools`.
2. Create a list of `Tool` in the file. For Example:

```python
from langchain.agents import Tool


def simple_calc(q):
    q = q.split(' ')
    if len(q) != 3:
        return 'Invalid input'
    try:
        a = float(q[0])
        b = float(q[2])
    except ValueError:
        return 'Invalid input'
    if q[1] == '+':
        return str(a + b)
    elif q[1] == '-':
        return str(a - b)
    elif q[1] == '*':
        return str(a * b)
    elif q[1] == '/':
        return str(a / b)


TOOLS = [
        Tool(
                name="my_calculator",
                description="This tool can be used to do simple calculations. The only supported operations are +, -, "
                            "*, and /. The input should be in the form of 'a + b', where a and b are numbers.",
                func=lambda q: simple_calc(q),
                )

        ]

```

3. Add your new set of tools to the root of the config file you want to use it. It must follow the format of this example:

```yaml
custom_tools:
  package: my_code.my_tools
  attribute: TOOLS
```

### Tweaking experts and tools

```bash
cp ./shared/experts_gpt.yaml ./my_code/experts_gpt.yaml
```

You copy the prompts file and point `PROMPTS_FILE_PATH` env to your new file.
