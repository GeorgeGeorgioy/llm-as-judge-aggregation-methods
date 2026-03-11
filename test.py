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
      cmd = input("> ").strip().lower()  

      if cmd == "oneshot":


        model = VLLMOnlineModel(opt, model_id, alias)
        model.run()  

      elif cmd == "stop":
        server.stop_server()
        break
      else:
        print("Unknown command. Type 'oneshot' to run the model once, or 'stop' to stop the server and exit.")  

    
       
    








    """
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        """