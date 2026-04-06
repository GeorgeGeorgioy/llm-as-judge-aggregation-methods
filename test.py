"""General-purpose test script for LLM as a judge.

"""

import os
from pathlib import Path
from options.test_options import TestOptions
import torch
from servers.vllm_server import VLLMServerManager
from models.vllm_online_models import VLLMOnlineModel
from models.registry import resolve_model_id , normalize_model_alias
#from models import MODEL_RUNNERS

# try:
#     import wandb
# except ImportError:
#     print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

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

    opt.num_runs = 1
    opt.aggregation_method = "oneshot"

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

    if role == "generator":
        # π.χ. generator_BiasBio → BiasBio
        if parts[0] == "generator":
            parts = parts[1:]
        return "_".join(parts)

    elif role == "judge":
        # ex. generator_llama8_oneshot_Arena_judge
        # dataset should be always before  "judge"
        if parts[-1] == "judge":
            return parts[-3]
        else:
            raise ValueError(f"Unexpected judge filename format: {filename}")

    else:
        raise ValueError(f"Unknown role: {role}")

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    #print(vars(opt))
    model_id, alias = resolve_model_id(opt.model_name)  # opt.model = alias
    
    """
    From model name finds the actual model id that vLLM server needs to start with. For example, "qwen7" -> "Qwen/Qwen2.5-7B-Instruct". This is done because the model name provided by user via CLI may be different from the one used by vLLM server. The mapping is defined in models/registry.py.
    """
    server = VLLMServerManager(opt, model_id)
    server.start_server()
    print("Server is up:", server.base_url)


    print("Type 'oneshot' to run the model once, or 'stop' to stop the server and exit.")
    print("Usage: start_server <model_name> [generator|judge]")
    while True:
      cmd = input("> ").strip()  



      if cmd.startswith("oneshot_all"):
        """
        Type oneshot_all <folder_path> to run oneshot on all .jsonl files in a folder
        """
        folder_path = cmd.split(maxsplit=1)[1] if len(cmd.split(maxsplit=1)) > 1 else opt.promptroot
        run_oneshot_for_all_files(opt, model_id, alias, folder_path)

      elif cmd.startswith("oneshot"):
        """
        Type oneshot <promptroot_path> to test multiply prompts datasets
        """
        opt.num_runs= 1
        opt.aggregation_method= "oneshot"

        new_dataroot = cmd.split(maxsplit=1)[1] if len(cmd.split(maxsplit=1)) > 1 else opt.promptroot
        opt.promptroot = new_dataroot
        opt.dataset_name = extract_dataset_name(opt.promptroot, opt.role)
        print(f"[INFO] Using dataset: {opt.dshutdownataset_name}")



        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()  
  

      elif cmd.startswith("multirun"):  
        print("Good")
        opt.num_runs= 3
        opt.aggregation_method= "multirun"

        new_dataroot = cmd.split(maxsplit=1)[1] if len(cmd.split(maxsplit=1)) > 1 else opt.promptroot
        opt.promptroot = new_dataroot
        opt.dataset_name = extract_dataset_name(opt.promptroot,opt.role)
        print(f"[INFO] Using dataset: {opt.dataset_name}")

          
        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()

      elif cmd.startswith("stop_server"):
        server.stop_server()
        print("The server is down you have to start an otherone")
        print("give model name to start a new one or shutdown ")

      elif cmd.startswith("start_server"):  
        
        parts = cmd.strip().split()

        if len(parts) >= 2:
          opt.model_name = parts[1]

        if len(parts) >= 3:
          opt.role = parts[2]

        model_id, alias = resolve_model_id(opt.model_name)

        server = VLLMServerManager(opt, model_id)
        server.start_server()
        print("Server is up:", server.base_url)
        print("Model:", opt.model_name)
        print("Role:", opt.role)
        print("Server is up:", server.base_url)

      elif cmd == "shutdown":
        server.stop_server()
        break
      else:
        print("Unknown command. Type 'oneshot' to run the model once, or 'stop' to stop the server and exit.")  

    
       
    





