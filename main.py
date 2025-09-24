import json

with open("./db_schema.json", "r") as f:
  input_schema = json.load(f)


base_prompt = f"""
You are an expert SQL generator.

## Instructions
- Only generate a valid SQL query.
- Always end the query with a semicolon (;).
- Do not explain, repeat the question, or output anything except SQL.
- Use the database schema below.
- The list of valid property names is given below (from artifactproperty.name and executionproperty.name). 
- When user uses natural language (like "validation loss"), map it to the closest valid property name (e.g., 'val_loss').
- Never invent property names that are not in the list.

## Database Schema
{json.dumps(input_schema, indent=2)}

## Valid Property Names
ArtifactProperty:
labels, url, accuracy, model_type, user-metadata1, labels_uri, git_repo,
roc_auc, user1, metrics_name, original_create_time_since_epoch, dataset_uri,
model_name, model_framework, Commit, val_loss, avg_prec

ExecutionProperty:
seed, split, input_signals, test_percent, Python_Env, train_percent, n_est,
ngrams, Git_Start_Commit, output_classes, Git_End_Commit, Execution,
original_create_time_since_epoch, Execution_type_name, Pipeline_id,
training_files, test_files, Git_Repo, Execution_uuid, Context_Type,
max_features, min_split, Pipeline_Type, Context_ID

## Examples
nlp: Which artifacts were linked with GitHub repository `mode_classifier`?
sql: SELECT a.*
FROM artifact a
JOIN artifactproperty ap ON a.id = ap.artifact_id
WHERE ap.name = 'git_repo'
  AND ap.string_value LIKE '%mode_classifier%';

nlp: I’d like to see all the execution runs under the pipeline `Training-percent_3`.
sql: SELECT e.*
FROM execution e
JOIN executionproperty ep ON e.id = ep.execution_id
WHERE ep.name = 'Pipeline_Type'
  AND ep.string_value = 'Training-percent_3';

nlp: Show all artifact names containing `cnn_lh_predictor`.
sql: SELECT id, name
FROM artifact
WHERE name LIKE '%cnn_lh_predictor%';

nlp: Display all artifacts whose model_name, model_framework, and model_type are null
sql: SELECT a.*
FROM artifact a
WHERE NOT EXISTS (
    SELECT 1 FROM artifactproperty ap
    WHERE ap.artifact_id = a.id AND ap.name = 'model_name'
)
AND NOT EXISTS (
    SELECT 1 FROM artifactproperty ap
    WHERE ap.artifact_id = a.id AND ap.name = 'model_framework'
)
AND NOT EXISTS (
    SELECT 1 FROM artifactproperty ap
    WHERE ap.artifact_id = a.id AND ap.name = 'model_type'
);

nlp: Which model has `model_name` set to `TCNN1d`?
sql: SELECT a.id AS artifact_id,
       a.name AS artifact_name,
       ap.string_value AS model_name
FROM artifact a
JOIN artifactproperty ap 
     ON a.id = ap.artifact_id
WHERE ap.name = 'model_name'
  AND ap.string_value = 'TCNN1d';

nlp: Display all executions whose train percent is 0.9.
sql: SELECT e.id AS execution_id,
       e.name AS execution_name,
       ep.double_value AS train_percent
FROM execution e
JOIN executionproperty ep 
     ON e.id = ep.execution_id
WHERE ep.name = 'train_percent'
  AND ep.double_value = 0.9;

nlp: Display dataslice with name `cmf_artifacts/3b75a834-611b-11ef-9f7e-a4bf0103caf6/dataslice/training_0.5:880227273ac75a7ae2341239c67134cd`
sql: SELECT a.*
FROM artifact a
JOIN type t 
     ON a.type_id = t.id
WHERE t.name = 'Dataslice'
  AND a.name = 'cmf_artifacts/3b75a834-611b-11ef-9f7e-a4bf0103caf6/dataslice/training_0.5:880227273ac75a7ae2341239c67134cd';

nlp: Display all artifacts created on 09 Sep 2025.
sql: SELECT a.*
FROM artifact a
WHERE TO_CHAR(TO_TIMESTAMP(a.create_time_since_epoch / 1000), 'YYYY-MM-DD') = '2025-09-09';

nlp: Display all executions containing output classes ['BBQH', 'QH', 'WPQH'].
sql: SELECT e.*
FROM execution e
JOIN executionproperty ep 
     ON e.id = ep.execution_id
WHERE ep.name = 'output_classes'
  AND ep.string_value = '[''BBQH'', ''QH'', ''WPQH'']';

nlp: Display all executions using fewer than 40 training files.
sql: SELECT e.*
FROM execution e
JOIN executionproperty ep
     ON e.id = ep.execution_id
WHERE ep.name = 'training_files'
  AND (
        LENGTH(ep.string_value) - LENGTH(REPLACE(ep.string_value, ',', '')) + 1
      ) < 40;

---

## Now generate SQL
nlp: {{user_question}}
sql:
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "./Qwen-2.5-3b-Text_to_SQL"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


def nlp_to_sql(question):
    # Add clear formatting
    prompt = base_prompt.strip() + f"\n\nnlp: {question}\nsql:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,        # prevent overly long rambling
        do_sample=False,           # deterministic, no random sampling
        eos_token_id=tokenizer.eos_token_id  # stop at end-of-sequence
    )

    # Only decode *new tokens after the prompt*
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    # Clean output
    result = result.strip()
    if ";" in result:   # cut off anything after first semicolon
        result = result.split(";")[0].strip() + ";"

    return result


if __name__ == "__main__":
  print("Enter your NLP queston (press Ctrl+C to exit):")
  try:
    while True:
      question = input("\nnlp:")
      sql = nlp_to_sql(question)
      print("sql:",sql)
  except KeyboardInterrupt:
    print("\nExiting...")

