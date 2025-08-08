Pyleste Environment - https://github.com/CelesteClassic/Pyleste

This python implementation of Celeste Classic was used and adapted to train our agent to play the game of Celeste. It is a perfect emulation of Celeste Classic (https://www.lexaloffle.com/bbs/?tid=2145) in Python.

The agent is currently set to train for 100000 episodes. At certain episodes, a list of actions is output as well as the total reward for that episode. These lists of actions were then used as inputs into a Tool Assisted Speedrunning (TAS) tool to perfectly transfer the agent's inputs frame-by-frame from training into the game itself.

In order to run the program you will need to install PyTorch and Matplotlib. Pytorch was to create the neural network, do backpropagation on it, and to turn our inputs into tensors to pass to the neural network. Matplotlib was used in order to plot a learning curve. Once these are installed all you need to do is run the program as normal.