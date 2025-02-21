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

import numpy as np
import time

from og_marl.loggers import JsonWriter

class BaseMARLSystem:
    def __init__(
        self,
        environment,
        logger,
        discount=0.99,
        add_agent_id_to_obs=False,
    ):

        self._environment = environment
        self._agents = environment.possible_agents
        self._logger = logger
        self._discount = discount
        self._add_agent_id_to_obs = add_agent_id_to_obs

        self._env_step_ctr = 0.0

    def get_stats(self):
        return {}

    def evaluate(self, num_eval_episodes=4):
        """Method to evaluate the system online (i.e. in the environment)."""
        episode_returns = []
        for episode in range(num_eval_episodes):
            self.reset()
            observations = self._environment.reset()

            if isinstance(observations, tuple):
                observations, infos = observations
            else:
                infos = {}

            done = False
            episode_return = 0
            while not done:
                
                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                actions = self.select_actions(observations, legal_actions, explore=False)
                
                observations, rewards, terminals, truncations, infos = self._environment.step(actions)
                episode_return += np.mean(list(rewards.values()))
                done = all(terminals.values()) or all(truncations.values())
            episode_returns.append(episode_return)
        logs = {"evaluator/episode_return": np.mean(episode_returns)}
        return logs
    
    def train_online(self, replay_buffer, max_env_steps=1e6, train_period=20):
        """Method to train the system online."""
        episodes = 0
        while True: # breaks out when env_steps > max_env_steps
            self.reset() # reset the system
            observations = self._environment.reset()

            if isinstance(observations, tuple):
                observations, infos = observations
            else:
                infos = {}
            
            episode_return = 0
            while True:
                if "legals" in infos:
                    legal_actions = infos["legals"]
                else:
                    legal_actions = None

                start_time = time.time()
                actions = self.select_actions(observations, legal_actions)
                end_time = time.time()
                time_for_action_selection = (end_time - start_time)

                start_time = time.time()
                next_observations, rewards, terminals, truncations, next_infos = self._environment.step(actions)
                end_time = time.time()
                time_to_step = (end_time - start_time)
                
                # Add step to replay buffer
                replay_buffer.add(observations, actions, rewards, terminals, truncations, infos)

                # Critical!!
                observations = next_observations
                infos = next_infos

                # Bookkeeping
                episode_return += np.mean(list(rewards.values()))
                self._env_step_ctr  += 1

                if self._env_step_ctr > 100 and self._env_step_ctr % train_period == 0: # TODO burn in period
                    # Sample replay buffer
                    start_time = time.time()
                    batch = next(replay_buffer)
                    end_time = time.time()
                    time_to_sample = (end_time - start_time)

                    # Train step
                    start_time = time.time()
                    train_logs = self.train_step(batch)
                    end_time = time.time()
                    time_train_step = (end_time - start_time)

                    train_steps_per_second = 1 / (time_train_step + time_to_sample)
                    env_steps_per_second = 1 / (time_to_step + time_for_action_selection)

                    train_logs = {**train_logs, **self.get_stats(), "Environment Steps": self._env_step_ctr, "Time to Sample": time_to_sample, "Time for Action Selection": time_for_action_selection, "Time to Step Env": time_to_step, "Time for Train Step": time_train_step, 
                                  "Train Steps Per Second": train_steps_per_second,
                                  "Env Steps Per Second": env_steps_per_second}

                    self._logger.write(train_logs)

                if all(terminals.values()) or all(truncations.values()):
                    replay_buffer.end_of_episode()
                    break

            episodes += 1
            if episodes % 20 == 0: # TODO: make variable
                self._logger.write({"Episodes": episodes, "Episode Return": episode_return, "Environment Steps": self._env_step_ctr}, force=True)

            if self._env_step_ctr > max_env_steps:
                break


    def train_offline(self, batched_dataset, max_trainer_steps=1e5, evaluate_every=1000, num_eval_episodes=4, json_writer=None):
        """Method to train the system offline.
        
        WARNING: make sure evaluate_every % log_every == 0 and log_every < evaluate_every, else you wont log evaluation.
        """
        trainer_step_ctr = 0
        while trainer_step_ctr < max_trainer_steps:

            if evaluate_every is not None and trainer_step_ctr % evaluate_every == 0:
                print("EVALUATION")
                eval_logs = self.evaluate(num_eval_episodes)
                self._logger.write(eval_logs, force=True)
                if json_writer is not None:
                    json_writer.write(
                        trainer_step_ctr,
                        "evaluator/episode_return",
                        eval_logs["evaluator/episode_return"],
                        trainer_step_ctr // evaluate_every
                    )
            
            start_time = time.time()
            batch = next(batched_dataset)
            end_time = time.time()
            time_to_sample = (end_time - start_time)

            start_time = time.time()
            train_logs = self.train_step(batch)
            end_time = time.time()
            time_train_step = (end_time - start_time)
            
            train_steps_per_second = 1 / (time_train_step + time_to_sample)

            logs = {**train_logs, "Trainer Steps": trainer_step_ctr, "Time to Sample": time_to_sample, "Time for Train Step": time_train_step, "Train Steps Per Second": train_steps_per_second}
            
            self._logger.write(logs)

            trainer_step_ctr += 1

        print("FINAL EVALUATION")
        eval_logs = self.evaluate(num_eval_episodes)
        self._logger.write(eval_logs, force=True)
        if json_writer is not None:
            eval_logs = {f"absolute/{key.split('/')[1]}": value for key, value in eval_logs.items()}
            json_writer.write(
                trainer_step_ctr,
                "absolute/episode_return",
                eval_logs["absolute/episode_return"],
                trainer_step_ctr // evaluate_every
            )

    def reset():
        """Called at the start of each new episode."""
        return

    def select_actions(self, observations):
        raise NotImplementedError
    
    def train_step(self, batch):
        raise NotImplementedError
