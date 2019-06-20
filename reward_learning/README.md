Reward learning code makes use of the companion code of the following publication:  
E Bıyık, D Sadigh. **"[Batch Active Preference-Based Learning of Reward Functions](https://arxiv.org/abs/1810.04303)"**. *Conference on Robot Learning (CoRL)*, Zurich, Switzerland, Oct. 2018.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [MuJoCo 1.50](http://www.mujoco.org/index.html)
- [NumPy](https://www.numpy.org/)
- [OpenAI Gym](https://gym.openai.com)
- [pyglet](https://bitbucket.org/pyglet/pyglet/wiki/Home)
- PYMC
- [Scikit-learn](https://scikit-learn.org)
- [SciPy](https://www.scipy.org/)
- [theano](http://deeplearning.net/software/theano/)

## Running
Throughout this demo,
- [task_name] should be selected as one of the following: LDS, Driver, LunarLander, MountainCar, Swimmer, Tosser
- [method] should be selected as one of the following: nonbatch, greedy, medoids, boundary_medoids, successive_elimination, dpp, random
For the details and positive integer parameters K, N, M, b, B; we refer to the CoRL publication cited above.
You should run the codes in the following order:

### Sampling the input space
This is the preprocessing step, so you need to run it only once (subsequent runs will overwrite for each task). It is not interactive and necessary only if you will use batch active preference-based learning. For non-batch version and random querying, you can skip this step.

You simply run
```python
	python input_sampler.py [task_name] K
```
For quick (but highly suboptimal) results, we recommend K=1000. In the NeurIPS submission, we used K=100000. In CoRL paper, the authors used K=500000.

### Learning preference reward function
This is where the actual algorithms work. You can simply run
```python
	python run.py [task_name] [method] N M b
```
b is required only for batch active learning methods. We fixed B=20b. To change that simply go to demos.py and modify 11th line.
Note: N must be divisible by b.
After each query or batch, the user will be showed the w-vector learned up to that point. To understand what those values correspond to, one can check the 'Tasks' section of the CoRL publication cited above.

Currently, we set the code to generate a random reward function and then learn it using [method]. To make it interactive and visual, you should edit the simulations_utils.py (see line 14 of that file).

### Demonstration of learned parameters
This is just for demonstration purposes. run_optimizer.py starts with 3 parameter values. You can simply modify them to see optimized behavior for different tasks and different w-vectors.
