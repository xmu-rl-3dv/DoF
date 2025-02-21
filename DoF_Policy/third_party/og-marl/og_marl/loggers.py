# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import wandb
import time

class TerminalLogger:

    def __init__(
            self,
            log_every=2 # seconds
    ):
        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs, force=False):
        
        if time.time() - self._last_log > self._log_every or force:

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

class WandbLogger:
    def __init__(
        self,
        config={},
        project="default_project",
        notes="",
        tags=["default"],
        entity=None,
        log_every=2 # seconds
    ):
        wandb.init(project=project, notes=notes, tags=tags, entity=entity, config=config)

        self._log_every = log_every
        self._ctr = 0
        self._last_log = time.time()

    def write(self, logs, force=False):
        
        
        if time.time() - self._last_log > self._log_every or force:
            wandb.log(logs)

            for key, log in logs.items():
                print(f"{key}: {float(log)} |", end=" ")
            print()

            if not force:
                self._last_log = time.time()

        self._ctr += 1

    def close(self):
        wandb.finish()

class JsonWriter:
    """
    Writer to create json files for reporting experiment results according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-
    This writer was adapted from the implementation found in BenchMARL. For the original
    implementation please see https://tinyurl.com/2t6fy548

    Args:
        path (str): where to write the file
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): random seed of the experiment
    """

    def __init__(
        self,
        path: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
    ):
        self.path = path
        self.experiment_path = path
        os.makedirs(self.experiment_path, exist_ok=True)
        self.experiment_id = self.get_new_experiment_id()
        
        self.file_name = "metrics.json"
        self.run_data = {"absolute_metrics": {}}
        

        self.algorithm_name = algorithm_name
        self.task_name = task_name
        self.environment_name = environment_name
        self.seed = seed
        self.file_name = f"metrics_{self.experiment_id}.json"
        self.run_data = {"absolute_metrics": {}}
        self.init_run_data()

    # 初始化log记录方式       
    def init_run_data(self):
        # Initialize the run data structure
        self.data = {
            self.environment_name: {
                self.task_name: {
                    self.algorithm_name: {
                        f"seed_{self.seed}": self.run_data
                    }
                }
            }
        }
        with open(os.path.join(self.experiment_path, self.file_name), 'w') as f:
            json.dump(self.data, f, indent=4)
    # 增加新的试验记录，类似sacred记录方式
    def get_new_experiment_id(self):
        # Generate a new experiment ID
        existing_ids = [int(name.split('_')[1].split('.')[0]) for name in os.listdir(self.path) \
                        if name.startswith("metrics_") and name.endswith(".json")]
        new_id = max(existing_ids) + 1 if existing_ids else 1
        return new_id
    
    def write(
        self,
        timestep: int,
        key: str,
        value: float,
        evaluation_step = None,
    ) -> None:
        """
        Writes a step to the json reporting file

        Args:
            timestep (int): the current environment timestep
            key (str): the metric that should be logged
            value (str): the value of the metric that should be logged
            evaluation_step (int): the evaluation step
        """

        logging_prefix, *metric_key = key.split("/")
        metric_key = "/".join(metric_key)

        metrics = {metric_key: [value]}

        if logging_prefix == "evaluator":
            step_metrics = {"step_count": timestep}
            step_metrics.update(metrics)  # type: ignore
            step_str = f"step_{evaluation_step}"
            if step_str in self.run_data:
                self.run_data[step_str].update(step_metrics)
            else:
                self.run_data[step_str] = step_metrics

        # Store the absolute metrics
        if logging_prefix == "absolute":
            self.run_data["absolute_metrics"].update(metrics)

        with open(os.path.join(self.experiment_path, self.file_name), 'w') as f:
            json.dump(self.data, f, indent=4)
