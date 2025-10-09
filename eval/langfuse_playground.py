import json
import os
from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langfuse import observe, get_client, Langfuse
from langfuse.api.resources.commons.errors import NotFoundError
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

import search
import generate


load_dotenv()

langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST"),
)

#%%
with open("./eval/eval_config.json", "r") as file:
    eval_config = json.load(file)
experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_description = eval_config["experiment_description"]
df = pd.read_parquet(eval_config["eval_data_path"])

#%%
np.random.seed(142)
n_samples = eval_config["n_samples"]

if n_samples < len(df):
    sample_ids = np.random.choice(len(df), size=n_samples, replace=False)
    json_data = df.iloc[sample_ids].to_dict(orient='records')
else:
    json_data = df.to_dict(orient='records')

dataset_name = eval_config["langfuse_dataset_name"]

try:
    langfuse.get_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' found.")
except NotFoundError:
    print(f"Dataset '{dataset_name}' created.")
    langfuse.create_dataset(name=dataset_name)
    for item in json_data:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item['question'],
            expected_output=item['answer'],
            metadata=item
        )

#%%
with open("./user_config.json", "r") as file:
    user_config = json.load(file)

with open("./opt_config.json", "r") as file:
    opt_config = json.load(file)

langfuse = get_client()


@observe()
def run_qanda(query, user_config, opt_config):
    results = search.search(query, user_config, opt_config)
    metadata = {"search_results": {}}
    paths = ''
    content = ''

    for result in results:
        print(f"ID: {result.id}, Path: {result.payload['path']}, Score: {result.score}")
        paths = '\n'.join([paths, result.payload["path"]])
        content = '\n\n-----\n\n'.join([content, result.payload["content"]])

    metadata["search_results"]["paths"] = paths
    metadata["search_results"]["content"] = content

    answer = generate.generate(query, results, opt_config)
    print(answer)

    langfuse.update_current_trace(
        input=query,
        output=answer,
        metadata=metadata,
        name=experiment_name,
    )
    return answer, results


#%%
def run_qanda_experiment(experiment_name):
    dataset = langfuse.get_dataset(dataset_name)
    for item in dataset.items:
        print(item)
        with item.run(
            run_name=experiment_name,
            run_metadata=opt_config,
            run_description=experiment_description,
        ) as root_span:
            output = run_qanda(item.input, user_config, opt_config)

run_qanda_experiment(experiment_name=experiment_name)
langfuse.flush()
os.sync()
sleep(15)
langfuse.flush()

# %%
# fetch traces for specific experiment
traces_batch = langfuse.api.trace.list(name=experiment_name, limit=n_samples).data

print(f"Traces in batch: {len(traces_batch)}")

# %%
# set local model in CLI:
# deepeval set-ollama 'llama3.3:70b'

# update traces with deepeval metrices
def run_deepeval_experiment():
    dataset = langfuse.get_dataset(dataset_name)
    for item in traces_batch:
        dataset_item_id = item.metadata["dataset_item_id"]
        dataset_item = next((item for item in dataset.items if item.id == dataset_item_id), None)
        if dataset_item:
            expected_output = dataset_item.expected_output
            print(item)
            test_case = LLMTestCase(
                input=item.input,
                actual_output=item.output,
                expected_output=expected_output,
                retrieval_context=item.metadata['search_results']['content'].split('\n\n-----\n\n')[1:]
            )

            ffm = FaithfulnessMetric(threshold=0.75,model=None)
            ffm.measure(test_case)
            print('- FFM Score: ', ffm.score)

            langfuse.create_score(
                trace_id=item.id,
                name="FFM",
                value=ffm.score,
                comment=ffm.reason
            )

            arm = AnswerRelevancyMetric(threshold=0.7, model=None)
            arm.measure(test_case)
            print("- ARM Score: ", arm.score)

            langfuse.create_score(
                trace_id=item.id,
                name="ARM",
                value=arm.score,
                comment=arm.reason
            )

run_deepeval_experiment()
langfuse.flush()