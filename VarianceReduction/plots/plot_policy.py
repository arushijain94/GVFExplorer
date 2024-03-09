import matplotlib.pyplot as plt
import numpy as np

# Define the number of rows and columns in the grid
num_rows = 4
num_cols = 4

# Define the policy probabilities for each state-action pair
policy_probs = np.random.rand(num_rows, num_cols, 4)  # Assuming 4 possible actions per state
print("pol:", policy_probs)

# Create a binary mask to represent blocked states (0 for blocked, 1 for existing)
state_mask = np.array([[1, 1, 1, 0],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [0, 1, 1, 1]])

# Create a grid with zeros, where each cell will represent the action with the maximum probability
max_action_grid = np.zeros((num_rows, num_cols), dtype=int)

# Loop through each state and find the action with the maximum probability
for row in range(num_rows):
    for col in range(num_cols):
        max_action_grid[row, col] = np.argmax(policy_probs[row, col])

# Create a grid of arrows to represent the actions
arrow_grid = np.empty((num_rows, num_cols), dtype=str)
arrow_grid.fill('')

# Define arrow directions for up, down, left, and right
arrows = {
    0: '↑',  # Up
    1: '↓',  # Down
    2: '←',  # Left
    3: '→'  # Right
}

# Fill the arrow grid with arrow characters based on the maximum action
for row in range(num_rows):
    for col in range(num_cols):
        max_action = max_action_grid[row, col]
        arrow_grid[row, col] = arrows[max_action]

# Create a figure and plot the arrows and state colors
fig, ax = plt.subplots()
ax.axis('off')  # Turn off axis labels and ticks

# Plot the arrow grid with arrows for the maximum action
for row in range(num_rows):
    for col in range(num_cols):
        # Set the color based on the state_mask
        if state_mask[row, col] == 0:
            color = 'black'  # Blocked state
        else:
            color = 'blue'  # Existing state

        ax.annotate(arrow_grid[row, col], (col + 0.5, num_rows - row - 0.5), fontsize=16, ha='center', va='center',
                    color=color)

# Show the plot
plt.show()
