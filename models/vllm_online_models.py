
import os
import json
import requests

class VLLMOnlineModel:
    def __init__(self, opt, model_id, alias):
        self.opt = opt
        self.model_id = model_id
        self.alias = alias


    
    def run(self)-> str:
        print(f"I am in the {self.alias} model runner")

        url = f"http://{self.opt.host}:{self.opt.port}/v1/chat/completions"

        out_dir = os.path.join(self.opt.results_dir, self.opt.role, self.alias)
        os.makedirs(out_dir, exist_ok=True)
        out_filename = f"{self.alias}_{self.opt.role}_results.jsonl"
        out_path = os.path.join(out_dir, out_filename)

        with open(self.opt.promptroot, "r", encoding="utf-8") as f_in, \
            open(out_path, "w", encoding="utf-8") as f_out:

            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                ex_id = item.get("id", str(i))

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
                        "finish_reason": finish_reason,
                    }

                except Exception as e:
                    record = {
                        "id": ex_id,
                        "error": repr(e),
                        "ground_truth": item.get("ground_truth"),
                    }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if self.opt.test_limit > 0 and i >= self.opt.test_limit - 1:
                    break

        print("Saved results to:", out_path)
        return out_path