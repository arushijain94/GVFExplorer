import os
import json
from pathlib import Path
# pendulum
# result_loc = "/home/mila/j/jainarus/scratch/clean-rl/single-reward-dqn/DiscreteActionsPendulumEnv-v0__goal_0.78__goal0.78"
# result_loc = "/home/mila/j/jainarus/scratch/clean-rl/single-reward-dqn/DiscreteActionsPendulumEnv-v0__goal0__goal0.0"

# puddle
# result_loc = "/home/mila/j/jainarus/scratch/clean-rl/puddle/single-reward-dqn/PuddleMultiGoals-v0__goal_0__goal0.0__puddle0"
# result_loc ="/home/mila/j/jainarus/scratch/clean-rl/puddle/single-reward-dqn/PuddleMultiGoals-v0__goal_1__goal0.0__puddle1"

# rooms
# result_loc1 = "/home/mila/j/jainarus/scratch/multi_room/goal_0"
# result_loc2 = "/home/mila/j/jainarus/scratch/multi_room/goal_1"
# result_locs = ["/home/mila/j/jainarus/scratch/multi_room/goal_2"]

# multi rooms gvf
result_loc1 = "/home/mila/j/jainarus/scratch/multi_room/FR_0"
result_loc2 = "/home/mila/j/jainarus/scratch/multi_room/FR_1"
result_loc3 = "/home/mila/j/jainarus/scratch/multi_room/FR_2"

result_locs = [result_loc1, result_loc2, result_loc3]

def aggregate_results(folder_path):
    # Define the path to the folder containing the JSON files
    folder = Path(folder_path)

    # Initialize an empty dictionary to store aggregated results
    aggregated_results = {}
    num_states_read = 0
    # Loop through each file in the directory
    for file_path in folder.glob('*.json'):
        # Open and read the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Check if 'n_eps' is greater than 30000
            if data.get('n_eps', 0) > 180000:
                num_states_read +=1
                # Extract the 'state' and 'val' and add them to the aggregated results
                # Convert 'state' to a tuple so that it can be used as a dictionary key
                state_tuple = tuple(data['state'])
                # If the state is already in the results, we might want to update the value
                # based on some condition or policy. For now, we'll just overwrite.
                aggregated_results[state_tuple] = data['val']

    print("num states read:", num_states_read)
    return aggregated_results

def save_aggregated_results_to_file(aggregated_results, output_file):
    save_loc = os.path.join(output_file, "state_to_val.json")
    state_to_val_str_keys = {str(k): v for k, v in aggregated_results.items()}

    # Save the aggregated results to a JSON file
    with open(save_loc, 'w') as file:
        json.dump(state_to_val_str_keys, file, indent=4)

if __name__ == "__main__":
    # input_loc = os.path.join(result_loc, "true_val_pi_states")
    for result_loc in result_locs:
        input_loc = os.path.join(result_loc, "true_val_random_states")
        output_loc = os.path.join(result_loc, "combined_true_val_random_states")
        if not os.path.exists(output_loc):
            os.makedirs(output_loc)

        aggregated_results = aggregate_results(input_loc)
        save_aggregated_results_to_file(aggregated_results, output_loc)
        print("done!")

