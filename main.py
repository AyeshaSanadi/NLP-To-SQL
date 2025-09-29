import json

with open("./db_schema.json", "r") as f:
  input_schema = json.load(f)

with open("./few-shot-questions.txt", "r") as file:
  few_shot_examples = file.read()

base_prompt = f"""
You are an expert SQL generator.

## Instructions

- Only generate a valid SQL query.
- Always end the query with a semicolon (;).
- Do not explain, repeat the question, or output anything except SQL.
- Use the database schema provided below.
- The properties related to artifacts and executions are stored inside `artifactproperty`
  and `executionproperty` tables under the `name` column respectivly.
- The properties which are available in `name` columns are the following:

  **ArtifactProperty:**
  labels, url, accuracy, model_type, user-metadata1, labels_uri, git_repo,
  roc_auc, user1, metrics_name, original_create_time_since_epoch, dataset_uri,
  model_name, model_framework, Commit, val_loss, avg_prec

  **ExecutionProperty:**
  seed, split, input_signals, test_percent, Python_Env, train_percent, n_est,
  ngrams, Git_Start_Commit, output_classes, Git_End_Commit, Execution,
  original_create_time_since_epoch, Execution_type_name, Pipeline_id,
  training_files, test_files, Git_Repo, Execution_uuid, Context_Type,
  max_features, min_split, Pipeline_Type, Context_ID

- When the user uses natural language terms (e.g., "validation loss", "github repo"), map them to the closest valid property name (e.g., "val_loss", "git_repo").
- Never invent property names that are not in the list above.

## Database Schema

Here are the database schemas, including the description of each table, table schema, indexes, and 10 sample records in JSON format:
{json.dumps(input_schema, indent=2)}

## Examples Context

The following examples are provided for reference.  
Each example demonstrates how to map a user’s natural language question into a valid SQL query using the given schema and property rules.

- User Question: The natural language question asked by the user.
- Expected SQL: The valid SQL query that answers the user question. The SQL query will always be enclosed inside `sql ... ` fences.
- Description: An explanation of how the SQL query was generated and why specific tables or columns were used.
- Note: Additional details about table relationships or important schema connections used in the query.

{few_shot_examples}

## Your task is to generate question and valid sql pair

User Question: {{user_question}}
Expected SQL:
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
    prompt = base_prompt.strip() + f"\n\nUser Question: {question}\nExpected SQL:"
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

  
def cleanup():
    # Delete model and tokenizer
    global model, tokenizer
    del model
    del tokenizer
    # Free GPU memory
    torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    print("Enter your NLP question (press Ctrl+C to exit):")
    user_questions = [
        "Which executions belong to context id `2`", 
        "display all the models that were created during the training stage?",
        "Can you show me all models where the validation loss dropped below 1.0?",
        "What's the accuracy of the very first model that was trained?",
        "list all the models with accuracy between 0.2 and 0.4?",
        "What's the earliest dataset recorded here?",
        "Which models specify `model_framework` as Pytorch?",
        "display all execution whose test percent is 0.9",
        "What was the lowest validation loss recorded, and which model does it belong to?",
        "Get artifacts created after timestamp 1757410442820.",
        "display artifacts whose original_create_time_since_epoch is 1724376071902",
        "Which model entries are associated with commit 40bcdcb75a2e86ad824e3032e516a4b3?",
        "Display dataset which has execution type name contains /Train",
        "fetch executions that have a property named pythonenv",
        "Display artifacts where artifact type is label",
        "display all executions where maximum features are used above 1k",
        "Fetch executions that have any of the three properties (training_files, ngrams, and split)",
        "display all artifacts where md5 keyword is present inside url.",
        "display artifacts whose average precision is above and equal to .5",
        "Display all the artifacts that contains less than 2 custom properties"
    ]

    try:
        for question in user_questions:
            # Safe Unicode printing
            safe_question = question.encode("utf-8", errors="replace").decode("utf-8")
            print("NLP Question:", safe_question)

            sql = nlp_to_sql(question)

            safe_sql = sql.encode("utf-8", errors="replace").decode("utf-8")
            print("SQL:", safe_sql)

    except KeyboardInterrupt:
        print("\nExiting...")

    finally:
        cleanup()  # Frees GPU memory
        print("GPU memory cleared and objects deleted.")


