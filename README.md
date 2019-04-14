# DRLND-Collaboration-and-Competition
Deep Reinforcement Learning Nanodegree Program

_The third project from Udacity Deep Reinforcement Learning Nanodegree_

# Multi-Agent Collaboration and Competition | RL Agents Play Tennis
##### &nbsp;
![Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/tennis.png?raw=true)
> [Figure 1:](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/images/tennis.png?raw=true) _Agents Playing Tennis_

## Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The MADDPG is implemented in Python 3 using PyTorch.

##### &nbsp;
## The Environment


For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md?fireglass_rsn=true#tennis&fireglass_params|&clear_tab_id=true&anti_bot_permission) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
 - After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
 - This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

**Note**: The project environment is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md?fireglass_rsn=true#fireglass_params|&clear_tab_id=true&anti_bot_permission).

##### &nbsp;
## The State and Action Spaces

```python
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: [ 0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.         -6.65278625 -1.5        -0.          0.
  6.83172083  6.         -0.          0.        ]
```

##### &nbsp;


## Repository Structure
The code is structured as follows:
* **Tennis.ipynb**: This is where the _Multi Agents_ are trained.
* **Tennis.html**: html view for Tennis.ipynb_
* **maddpg_agent.py**: This module implements MADDPG and represents a _Multi Agents_.
* **model.py**: This module contains the implementation of the _Actor and Critic_ neural networks.
* **actor_0.pth**: This is the binary containing the trained neural network weights for Actor 0.
* **actor_1.pth**: This is the binary containing the trained neural network weights for Actor 1.
* **critic_0.pth**: This is the binary containing the trained neural network weights for Critic 0.
* **critic_1.pth**: This is the binary containing the trained neural network weights for Critic 1.
* **Report.md**: Project report and result & test analysis.
* **README.md**: Readme file.



##### &nbsp;

## Dependencies

The overall dependencies are the following;
* python 3.6
* numpy: Install with 'pip install numpy'.
* PyTorch: Install by following the instructions [here](https://github.com/reinforcement-learning-kr/pg_travel/wiki/Installing-Unity-ml-agents-on-Windows).
* ml-agents: Install by following instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).

##### &nbsp;

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	activate drlnd
	```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

![Kernel](Images/Kernel.png)
> [Figure 2:](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png) _Kernel_

##### &nbsp;

## Instructions

Follow the instructions in `Tennis.ipynb` to get started with training your own agent!  
Trained model weights is included for quickly running the agent and seeing the result in Unity ML Agent.
- Run cell 11 in the notebook `Tennis.ipynb` to plot the scores for multi agents.
