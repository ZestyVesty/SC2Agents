# Error message from terminal, is it just out of memory on the GPU?
Global training episode: 216
Target network updated.
I0527 17:09:14.424153 140425068468032 sc2_env.py:632] Episode 216 finished after 2880 game steps. Outcome: [1], reward: [0], score: [6]
Model Saved
Summary Written
Game has started.
I0527 17:09:22.326003 140425068468032 sc2_env.py:462] Starting episode: 217
Global training episode: 217
Took 4682.171 seconds for 58355 steps: 12.463 fps
I0527 17:09:31.184732 140425068468032 sc2_env.py:656] Environment Close
RequestQuit command received.
Closing Application...
unable to parse websocket frame.
Terminate action already called.
Entering core terminate.
Core shutdown finished.
I0527 17:09:32.532938 140425068468032 sc_process.py:201] Shutdown gracefully.
I0527 17:09:32.535597 140425068468032 sc_process.py:182] Shutdown with return code: 0
Traceback (most recent call last):
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/home/sylvester/Desktop/honors/SC2Agents/run.py", line 170, in <module>
    app.run(main)
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/site-packages/absl/app.py", line 300, in run
    _run_main(main, args)
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "/home/sylvester/Desktop/honors/SC2Agents/run.py", line 156, in main
    run_thread(agent_classes, players, FLAGS.map, FLAGS.render)
  File "/home/sylvester/Desktop/honors/SC2Agents/run.py", line 119, in run_thread
    run_loop.run_loop(agents, env, FLAGS.max_agent_steps, FLAGS.max_episodes)
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/site-packages/pysc2/env/run_loop.py", line 43, in run_loop
    for agent, timestep in zip(agents, timesteps)]
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/site-packages/pysc2/env/run_loop.py", line 43, in <listcomp>
    for agent, timestep in zip(agents, timesteps)]
  File "/home/sylvester/Desktop/honors/SC2Agents/agents/dueling_DQN.py", line 196, in step
    self._train_network()
  File "/home/sylvester/Desktop/honors/SC2Agents/agents/dueling_DQN.py", line 292, in _train_network
    states, actions, targets = self._get_batch()
  File "/home/sylvester/Desktop/honors/SC2Agents/agents/dueling_DQN.py", line 304, in _get_batch
    actions = np.eye(np.prod(feature_screen_size))[actions]
  File "/home/sylvester/anaconda3/envs/tf-gpu/lib/python3.7/site-packages/numpy/lib/twodim_base.py", line 201, in eye
    m = zeros((N, M), dtype=dtype, order=order)
MemoryError
I0527 17:09:33.309409 140425068468032 sc2_env.py:656] Environment Close
I0527 17:09:33.309640 140425068468032 sc2_env.py:656] Environment Close


# Notes
Often get this error on the sick rig (alienware desktop with nvidia rtx 2080 Ti), but this error never happened on my
alienware laptop. It appears that the Ram in the Desktop keeps increasing bits by bits, but reason unknown.

Here are the differences:

- Alienware Desktop:
    * Uses GPU
    * Runs on python3.7
- Alienware Laptop:
    * Uses CPU (GPU only has 2G can't use it)
    * Runs on python3.6

Note: 
* Using different CUDA.
* runs the same code
# Observation
When running a training session on each of the machines, the ram usage for the Desktop keeps increasing. This can be
observed while running the program "htop" while the training session runs. Even though installing python 3.6 via conda
and run training with it doesn't fix the problem, usage of ram still increases.

The increase of ram usage cannot be found on the alienware laptop while training using python 3.6

# Experiment
- As descripbed above, tried to run the code on the Alienware Desktop with python 3.6, didn't change a thing.
- Tried to play around with the class Memory in the file dueling_DQN.py, seems like it stores a time step on a frequent
basis. If I continue the training of a model, all the previous stored data will be wiped out. In theory, if I'm correct
the size of the ram will increase as time pass by, but this isn't happening on the laptop.

# Some wild guesses (implying I don't know what im talking about)
- memory leak (get better RAMs Benny S.)
- Some python packages might be modified and it's doing something different to python3.6
- Garbage Collector is garbage and not doing it's job

# Note:
Currently training on the Alienware Desktop again, could of just been me ending it too early, will wait and see what happens tomorrow morning.
