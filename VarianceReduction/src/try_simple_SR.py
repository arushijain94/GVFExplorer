from envs.tabularMDP import TabularMDP

terminal_states=[(0,0)]
mdp = TabularMDP(terminal_states, size=5, gamma=0.99, env_name="simple_grid", stochastic_transition=False)
env = mdp.env
