"""General-purpose test script for LLM as a judge.

"""

import os
from pathlib import Path
from options.test_options import TestOptions
import torch
from servers.vllm_server import VLLMServerManager
from models.vllm_online_models import VLLMOnlineModel
from models.registry import resolve_model_id , normalize_model_alias
import time
import requests
#from models import MODEL_RUNNERS

# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def wait_for_vllm_server(base_url, check_interval=15, timeout=900):
    """
    Polls the vLLM server until it is ready.
    Checks every `check_interval` seconds.
    Stops after `timeout` seconds.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)

            if response.status_code == 200:
                print("[INFO] vLLM server is ready.")
                return

            print(f"[INFO] Server responded with status {response.status_code}. Waiting...")

        except requests.exceptions.RequestException:
            print("[INFO] Waiting for vLLM server...")

        time.sleep(check_interval)

    raise RuntimeError("vLLM server did not become ready within the timeout.")

def run_oneshot_for_all_files(opt, model_id, alias, folder_path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder does not exist: {folder_path}")
        return

    files = sorted(folder.glob("*.jsonl"))

    if not files:
        print(f"[WARNING] No .jsonl files found in: {folder_path}")
        return

    print(f"[INFO] Found {len(files)} files in {folder_path}")

    #opt.num_runs = 1
    #opt.aggregation_method = "oneshot"

    for file_path in files:

        
        print(f"\n[INFO] Running oneshot for: {file_path}")
        
        opt.promptroot = str(file_path)
        opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)
        print(f"[INFO] Using dataset: {opt.dataset_name}")

        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()
        

    print("\n[INFO] Finished processing all files.")


def extract_dataset_name(promptroot: str, role: str) -> str:
    filename = Path(promptroot).stem
    parts = filename.split("_")
    print(role)

    if role == "generator":
        # π.χ. generator_BiasBio → BiasBio
        if parts[0] == "generator":
            parts = parts[1:]
        return "_".join(parts)

    elif role == "judge":
        # ex. generator_llama8_oneshot_Arena_judge
        # dataset should be always before  "judge"
        if parts[-1] == "judge":
            print(parts[-3])
            return parts[-3]
        else:
            raise ValueError(f"Unexpected judge filename format: {filename}")

    else:
        raise ValueError(f"Unknown role: {role}")

if __name__ == "__main__":
    opt = TestOptions().parse()

    model_id, alias = resolve_model_id(opt.model_name)

    server = VLLMServerManager(opt, model_id)


    try:
        server.start_server()
        wait_for_vllm_server(server.base_url, check_interval=15, timeout=900)

        print("Server is up:", server.base_url)

        # ============================================================
        # ONESHOT
        # Τρέχει ένα συγκεκριμένο .jsonl αρχείο
        # ============================================================

         #opt.role = "generator"  # ή "judge"
        #opt.num_runs = 1
        #opt.aggregation_method = "oneshot"
        # opt.promptroot = "/path/to/your/file.jsonl"
        #opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        #print(f"[INFO] Running ONESHOT")
        #print(f"[INFO] Dataset: {opt.dataset_name}")


        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run()


        # ============================================================
        # ONESHOT ALL
        # Τρέχει oneshot για όλα τα .jsonl αρχεία σε έναν φάκελο
        # ============================================================

        # opt.role = "judge"  # ή "generator"
        # opt.num_runs = 1
        # opt.aggregation_method = "oneshot_all"

        # folder_path = "/path/to/your/folder"

        # print(f"[INFO] Running ONESHOT ALL")
        # print(f"[INFO] Folder: {folder_path}")

        # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # ============================================================
        # MULTIRUN_ALL
        # Τρέχει ένα συγκεκριμένο .jsonl αρχείο με 3 runs
        # ============================================================

        opt.role = "judge"
        opt.num_runs = 10

 

        opt.temperature = 0.25
        opt.aggregation_method = "multirun_0.25"

        # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_HaluEval_to_judge.jsonl"
        opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        model = VLLMOnlineModel(opt, model_id, alias)
        #model.run()


        # # opt.temperature = 0.5
        # # opt.aggregation_method = "multirun_0.5"

        # # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # # model = VLLMOnlineModel(opt, model_id, alias)
        # # model.run()


        # opt.temperature = 1
        # opt.aggregation_method = "multirun_1"

        # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run()        


        # # opt.temperature = 1
        # # opt.aggregation_method = "multirun_1"

        # # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # # model = VLLMOnlineModel(opt, model_id, alias)
        # # model.run()

        # opt.temperature = 1.5
        # opt.aggregation_method = "multirun_1.5"

        # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run() 


        # # opt.temperature = 1.5
        # # opt.aggregation_method = "multirun_1.5"

        # # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # # model = VLLMOnlineModel(opt, model_id, alias)
        # # model.run()


        # opt.temperature = 2
        # opt.aggregation_method = "multirun_2"

        # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run() 



        # # opt.temperature = 2
        # # opt.aggregation_method = "multirun_2"

        # # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # # model = VLLMOnlineModel(opt, model_id, alias)
        # # model.run()

        # opt.temperature = 2.5
        # opt.aggregation_method = "multirun_2.5" 
       

        # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run()

        # # opt.temperature = 2.5
        # # opt.aggregation_method = "multirun_2.5" 
       

        # # # folder_path = "/work3/s233559/Thesis/prompts/judge"
        # # # run_oneshot_for_all_files(opt, model_id, alias, folder_path)

        # # opt.promptroot = "/work3/s233559/Thesis/prompts/judge/generator_mistral7_oneshot_Arena_to_judge.jsonl"
        # # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # # model = VLLMOnlineModel(opt, model_id, alias)
        # # model.run()


        # #folder_path = "/work3/s233559/Thesis/prompts/judge"
        # #run_oneshot_for_all_files(opt, model_id, alias, folder_path)
        # ============================================================
        # MULTIRUN
        # Τρέχει ένα συγκεκριμένο .jsonl αρχείο με 3 runs
        # ============================================================

        # opt.role = "generator"  # ή "judge"
        # opt.num_runs = 3
        # opt.aggregation_method = "multirun"
        # opt.promptroot = "/path/to/your/file.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # print(f"[INFO] Running MULTIRUN")
        # print(f"[INFO] Dataset: {opt.dataset_name}")

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run()


        # ============================================================
        # CHANGE MODEL / ROLE EXAMPLE
        # Αν θες να αλλάξεις μοντέλο και role μέσα από το script
        # ============================================================

        # opt.model_name = "qwen7"
        # opt.role = "judge"

        # model_id, alias = resolve_model_id(opt.model_name)

        # opt.num_runs = 1
        # opt.aggregation_method = "oneshot"
        # opt.promptroot = "/path/to/your/file.jsonl"
        # opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)

        # model = VLLMOnlineModel(opt, model_id, alias)
        # model.run()

    finally:
        server.stop_server()
        print("[INFO] Server stopped.")