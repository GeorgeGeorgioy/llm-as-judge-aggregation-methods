
import os
import json
import requests

class VLLMOnlineModel:
    def __init__(self, opt, model_id, alias):
        self.opt = opt
        self.model_id = model_id
        self.alias = alias





    def end_point(self):
        return f"http://{self.opt.host}:{self.opt.port}/v1/chat/completions"   


    def pathes_diffinder(self):

        out_dir = os.path.join(self.opt.results_dir, self.opt.role, self.alias)
        os.makedirs(out_dir, exist_ok=True)
        prompt_path = self.opt.promptroot

        if self.opt.role == "judge":

            filename = os.path.basename(self.opt.promptroot)
            generator_name = filename.replace("from_", "").split("_judge")[0]

            out_filename = (
                f"{self.opt.role}_{self.alias}_generator_{generator_name}_"
                f"{self.opt.aggregation_method}_{self.opt.dataset_name}_results.jsonl"
            )

        elif self.opt.role == "generator":

            out_filename = (
                f"{self.opt.role}_{self.alias}_"
                f"{self.opt.aggregation_method}_{self.opt.dataset_name}_results.jsonl"
            )

        else:
            raise ValueError(f"Unknown role: {self.opt.role}")

        out_path = os.path.join(out_dir, out_filename)

        return out_path, prompt_path


    def api_call(self, url,item, ex_id)-> dict:

        payload = {
            "model": self.model_id,
            "messages": item["messages"],
            "max_tokens": self.opt.max_tokens,
            "temperature": self.opt.temperature,
        }

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            prediction = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason")

            record = {
                "id": ex_id,
                "prediction": prediction,
                "ground_truth": item.get("ground_truth"),
                "generators_answer": item.get("metadata", {}).get("generators_answer"), 
                "finish_reason": finish_reason,
            }

        except Exception as e:
            record = {
                "id": ex_id,
                "error": repr(e),
                "ground_truth": item.get("ground_truth"),
                    }


        return record
        


    
    def run(self)-> str:
        print(f"I am in the {self.alias} model runner")

        # url = f"http://{self.opt.host}:{self.opt.port}/v1/chat/completions"
        url = self.end_point()

        # out_dir = os.path.join(self.opt.results_dir, self.opt.role, self.alias)
        # os.makedirs(out_dir, exist_ok=True)
        # out_filename = f"{self.alias}_{self.opt.role}_{self.opt.aggregation_method}_{self.opt.dataset_name}_results.jsonl"
        # out_path = os.path.join(out_dir, out_filename)

        out_path, prompt_path = self.pathes_diffinder()

        with open(prompt_path, "r", encoding="utf-8") as f_in, \
            open(out_path, "w", encoding="utf-8") as f_out:

            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                ex_id = item.get("id", str(i))

                runs = 1
                if self.opt.aggregation_method == "multi-run":
                    runs = self.opt.num_runs

                for run_id in range(runs):

                    record = self.api_call(url, item ,ex_id)

                    #--------------------------------------------------------
                    
                    with open("raw_records.txt", "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    #--------------------------------------------------------    
                    record["run_id"] = run_id

                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if self.opt.test_limit > 0 and i >= self.opt.test_limit - 1:
                    break

        print("Saved results to:", out_path)
        return out_path