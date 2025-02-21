import os
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from og_marl.environments import smac, mamujoco
from og_marl.environments import simple_spread

def main(env_name: str="mpe", map_name: str="simple_spread", quality: str="expert"):
    add_agent_id_to_obs = True
    dataset_dir = Path(f"diffuser/datasets/data/{env_name}/{map_name}/{quality}")

    file_path = Path(dataset_dir)
    sub_dir_to_idx = {}
    idx = 0
    for subdir in os.listdir(file_path):
        if file_path.joinpath(subdir).is_dir():
            sub_dir_to_idx[subdir] = idx
            idx += 1
    
    
    for subdir in os.listdir(file_path):
        filenames = [str(file_name) for file_name in file_path.joinpath(subdir).glob("*.npy")]
        # filenames = sorted(filenames, key=get_fname_idx)
        filenames = sorted(filenames)

        (
            all_observations,
            all_actions,
            all_rewards,
            all_discounts,
            all_logprobs,
            all_dones,
            all_next_observations
        ) = ([], [], [], [], [], [], [])

        for filename in filenames:
            data = np.load(filename)
            if "acs" in filename:
                all_actions.append(data)
            elif "dones" in filename:
                all_dones.append(data)
            elif "next_obs" in filename:
                all_next_observations.append(data)
            elif "obs" in filename:
                all_observations.append(data)
            elif "rews" in filename:
                all_rewards.append(data)
        
        num_agents = len(all_actions)
        assert len(all_dones) == num_agents
        assert len(all_next_observations) == num_agents
        assert len(all_observations) == num_agents
        assert len(all_rewards) == num_agents

        length = all_actions[0].shape[0]
        assert all_dones[0].shape[0] == length
        assert all_next_observations[0].shape[0] == length
        assert all_observations[0].shape[0] == length
        assert all_rewards[0].shape[0] == length

        def save_to_tfrecord(all_observations, all_actions, all_rewards, all_discounts, all_logprobs, all_dones, all_next_observations, output_file):
            for l in range(length):                
                with tf.io.TFRecordWriter(output_file + str(l // 200) + ".tfrecord", options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
                    feature = {}
                    for a in range(num_agents):
      
                        feature[f'agent_{a}_observations'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[all_observations[a][l].tobytes()]))
                        feature[f'agent_{a}_actions'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[all_actions[a][l].tobytes()]))
                        feature[f'agent_{a}_rewards'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[all_rewards[a][l].tobytes()]))
                        feature[f'agent_{a}_next_observations'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[all_next_observations[a][l].tobytes()]))
                        feature[f'agent_{a}_dones'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[all_dones[a][l].tobytes()]))
                        feature[f'agent_{a}_discounts'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.float32(0.99).tobytes()]))
                        feature[f'agent_{a}_legal_actions'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.ones(5, "float32").tobytes()]))
                    feature[f'env_state'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.zeros(54, "float32").tobytes()]))
                    feature[f'episode_return'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.zeros(1, "float32").tobytes()]))
                    feature[f'zero_padding_mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(1, dtype=np.float32).tobytes()]))

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                if l % 200 == 0:
                    print(f"Saved {l} records to {output_file + str(l // 200)}.tfrecord")

        prefix = f"tests/dataset/{map_name}/{subdir}/"

        if not os.path.exists(prefix):
            os.makedirs(prefix)
            print(f"Directory '{prefix}' created.")
        else:
            print(f"Directory '{prefix}' already exists.")

        output_file = prefix + f"{map_name}_"
        # output_file = prefix + f"{map_name}.tfrecord"

        save_to_tfrecord(all_observations, all_actions, all_rewards, 
                         all_discounts, all_logprobs, all_dones, all_next_observations, 
                         output_file)
        
        break


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="mpe")
    parser.add_argument("--map_name", type=str, default="simple_spread")
    parser.add_argument("--quality", type=str, default="expert")
    args = parser.parse_args()

    main(args.env_name, args.map_name, args.quality)