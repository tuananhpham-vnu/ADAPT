## Run tool use, memory retrieval, and reasoning to generate training data for planning and testing input for planner

from pathlib import Path
import time
import json

from agentdriver.main.language_agent import LanguageAgent
from agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY, FINETUNE_PLANNER_NAME, OPENAI_BASE_URL
import argparse
import openai
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from adapt_tracing import setup_logging, step as trace_step, flush as flush_traces, is_enabled
# openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY
openai.base_url = OPENAI_BASE_URL

# if __name__ == "__main__":

def main():
    setup_logging()
    is_enabled()  # in trạng thái Braintrust ngay đầu run
    arg_dict = {"idx": args.idx}
    data_path = Path('agentdriver/data/')
    split = 'val'
    language_agent = LanguageAgent(
        data_path, 
        split, 
        model_name="gpt-3.5-turbo-0613", 
        # model_name="gpt-3.5-turbo", 
        planner_model_name=FINETUNE_PLANNER_NAME, 
        finetune_cot=False, 
        verbose=False
    )

    current_time = time.strftime("%D:%H:%M")
    current_time = current_time.replace("/", "_")
    current_time = current_time.replace(":", "_")
    # save_path = Path("experiments") / Path(current_time)
    save_path = Path("result") / Path(current_time)
    save_path.mkdir(exist_ok=True, parents=True)
    with open("agentdriver/data/finetune/data_samples_val.json", "r") as f:
        data_samples = json.load(f)[100:350]
    
    with trace_step("agentdriver.run", type="task",
                    metadata={"split": split, "num_samples": len(data_samples),
                              "planner_model": FINETUNE_PLANNER_NAME}):
        planning_traj_dict = language_agent.inference_all(
            data_samples=data_samples, 
            data_path=Path(data_path) / Path(split), 
            save_path=save_path,
            args=arg_dict
        )

    flush_traces()
    

argparser = argparse.ArgumentParser(description="Run tool use, memory retrieval, and reasoning to generate training data for planning and testing input for planner")
argparser.add_argument("--idx", type=int, default=0)

args = argparser.parse_args()
main()