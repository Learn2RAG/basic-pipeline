import logging
import logging.config
import yaml
import json

import ingestion
import search
import generate


if __name__ == "__main__":
    logging.config.dictConfig(yaml.safe_load(open("logging.yaml").read()))

    with open("user_config.json", "r") as file:
        user_config = json.load(file)

    with open("opt_config.json", "r") as file:
        opt_config = json.load(file)

    ingestion.index(user_config, opt_config)

    query = "What approach did Arjun Singh's campaign use to respond to voters' concerns on social media platforms during the municipal elections in Delhi?"
    results = search.search(query, user_config, opt_config)
    for result in results:
        print(f"ID: {result.id}, Path: {result.payload['path']}, Score: {result.score}")

    answer = generate.generate(query, results)

    print(query)
    print(answer)
