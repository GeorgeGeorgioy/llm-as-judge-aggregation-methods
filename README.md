# LLM Evaluation Pipeline

![Pipeline](assets/pipeline.png)

This project implements a modular pipeline for evaluating LLMs using the LLM-as-a-Judge paradigm.

## Features
- Generator / Judge setup
- vLLM server-based inference
- Aggregation methods


## Usage Guidelines

This project is executed through the command line using the main script:
# LLM Evaluation Pipeline

![Pipeline](assets/pipeline.png)

This project implements a modular pipeline for evaluating LLMs using the LLM-as-a-Judge paradigm.

## Features
- Generator / Judge setup
- vLLM server-based inference
- Aggregation methods


## Usage Guidelines

This project is executed through the command line using the main script:

```bash
python test.py --promptroot <path_to_prompt> --model_name <model_alias> --results_dir <results_path> --role <generator|judge>

 ``` 
## Required Arguments

- `--promptroot`: Path to the predefined prompt set  
- `--model_name`: Model alias (the server must be running)  
- `--results_dir`: Directory where results will be stored  
- `--role`: Either `generator` or `judge`  

---

## Notes

- The server corresponding to `<model_alias>` must be running before execution.  
- The `<path_to_prompt>` should point to a predefined prompt set.  

---

## Recommended Setup

- If `--role` is `generator`:
  - `--promptroot` → `./prompts/generator`  
  - `--results_dir` → `./results/generator`  

- If `--role` is `judge`:
  - `--promptroot` → `./prompts/judge`  
  - `--results_dir` → `./results/judge`  

---

## CLI Commands

### `oneshot`

Runs the generator or judge on the dataset once.

```bash
oneshot <path_to_prompt>
```
for testing different dataset with the same model.

### `multirun`

```bash
multirun <path_to_prompt>
```
Runs the evaluation multiple times (default = 3) to assess model consistency. For testing multiply datasets the <path_to_prompt> can be follow multirun <path_to_prompt>
  #### Note that arguments like tempratur and random seed should ajust properly.


### ``stop_server``
``` bash
stop_server
```
 
Stops the current server/model, So another model can be tested. 



### `start_server`

``` bash
start_server <model_alias>
```

 Starts a new servet for the model <model_alias>. 



### ``shutdown``

``` bash
shutdown
```
Terminates the aplication 

---

### Summary

This pipeline enables systematic evaluation of LLMs by separating generation and judging, allowing reproducible and scalable experimentation.

```bash
python test.py --promptroot <path_to_prompt> --model_name <model_alias> --results_dir <results_path> --role <generator|judge>

 ``` 
## Required Arguments

- `--promptroot`: Path to the predefined prompt set  
- `--model_name`: Model alias (the server must be running)  
- `--results_dir`: Directory where results will be stored  
- `--role`: Either `generator` or `judge`  

---

## Notes

- The server corresponding to `<model_alias>` must be running before execution.  
- The `<path_to_prompt>` should point to a predefined prompt set.  

---

## Recommended Setup

- If `--role` is `generator`:
  - `--promptroot` → `./prompts/generator`  
  - `--results_dir` → `./results/generator`  

- If `--role` is `judge`:
  - `--promptroot` → `./prompts/judge`  
  - `--results_dir` → `./results/judge`  

---

## CLI Commands

### `oneshot`

Runs the generator or judge on the dataset once.

```bash
oneshot <path_to_prompt>
for testing different dataset with the same model.

multirun:
Runs the evaluation multiple times (default = 3) to assess model consistency. For testing multiply datasets the <path_to_prompt> can be follow multirun <path_to_prompt>


stop_server: Stops the current server/model, So another model can be tested. 



start_server:
 Starts a new servet by : start_server <model_alias>



shutdown: Terminates the aplication 

---

## Summary

This pipeline enables systematic evaluation of LLMs by separating generation and judging, allowing reproducible and scalable experimentation.
