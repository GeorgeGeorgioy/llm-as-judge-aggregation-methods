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
    while True:
      cmd = input("> ").strip()  

      if cmd.startswith("oneshot"):
        new_dataroot = cmd.split(maxsplit=1)[1] if len(cmd.split(maxsplit=1)) > 1 else opt.promptroot
        opt.promptroot = new_dataroot

        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()  

      elif cmd == "multirun":  
          
        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()

      elif cmd.startswith("stop_server"):
        server.stop_server()
        print("The server is down you have to start an otherone")
        print("give model name to start a new one or shutdown ")
      elif cmd.startswith("start_server"):  
        
        new_model_name = cmd.split(maxsplit=1)[1] if len(cmd.split(maxsplit=1)) > 1 else opt.model_name
        opt.model_name = new_model_name


        model_id, alias = resolve_model_id(opt.model_name)  # opt.model = alias
    

        server = VLLMServerManager(opt, model_id)
        server.start_server()
        print("Server is up:", server.base_url)

      elif cmd == "shutdown":
        server.stop_server()
        break
      else:
        print("Unknown command. Type 'oneshot' to run the model once, or 'stop' to stop the server and exit.")  

    
       
    





