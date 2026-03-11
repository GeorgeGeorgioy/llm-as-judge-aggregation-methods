# Thesis/servers/vllm_server.py

import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional
import os
import sys
import collections
from pathlib import Path
from datetime import datetime


class VLLMServerManager:

    def __init__(self, opt, alias ) :

        
        self.host = opt.host
        self.port = int(opt.port)
        self.gpu = opt.gpu
        self.base_url = f"http://{self.host}:{self.port}"
        self.gpu_memory_utilization = opt.gpu_memory_utilization
        self.dtype = opt.dtype
        self.model_name = alias
        self.process = None
        self.opt = opt




    def health_check(self) -> bool:
        url = f"{self.base_url}/v1/models"

        try:
            with urllib.request.urlopen(url, timeout=2) as response:
             return response.status == 200
        except Exception:
            return False    


    def start_server(self):

       
    # --- change env variables if their is need ---
        env = os.environ.copy()

        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            env["VLLM_LOGGING_LEVEL"] = "DEBUG"
            env["VLLM_LOG_STATS_INTERVAL"] = "1"

    # prepare log file

        Path(self.opt.checkpoints_dir) / self.opt.experiment_name
        run_id = self.opt.model_name + "_" + datetime.now().strftime("%H%M%S")
        run_dir = Path(self.opt.checkpoints_dir) / self.opt.experiment_name / run_id

        os.makedirs(run_dir, exist_ok=True)

    # log file
        log_path = run_dir / "server.log"
        log_file = open(log_path, "a", buffering=1)

    
    # Starting the subprosess
        cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host", self.host,
        "--port", str(self.port),
        "--model", self.model_name,
        "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        "--max-model-len", str(self.opt.max_model_len),


    ]
      
        print("Starting vLLM server")

        self.process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        text=True,
        bufsize=1,
    )
      
        # logging output
        #last_lines = collections.deque(maxlen=200)

        """
        Needs more work for something better.

        """
        timeout = 120
        deadline = time.time() + timeout

        while time.time() < deadline:
            if self.health_check():
                print("Server is ready.")
                return

            # #if self.process.stdout is not None:
            #     # read lines until the process is ready or crashes
           
            time.sleep(0.5)

        raise RuntimeError("Server did not become ready in time.")
    
    
    #
    def stop_server(self):
        print("Stopping server...")

        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("Server stopped gracefully.")
            except subprocess.TimeoutExpired:
                print("Server did not stop in time. Killing it.")
                self.process.kill()
                self.process.wait()
                print("Server killed.")