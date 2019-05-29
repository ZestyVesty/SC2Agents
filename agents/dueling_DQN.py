"""Deep Q-learning agents."""



"""
Note: improve the input of the neural network by setting weights of not usable actions to 0 (for this environment)
"""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.value_estimators_dueling as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS

dq_mg = "DDQN_CMS"      # Type of Deep Q Learning and mini game name

if not os.path.exists("checkpoints/"+ dq_mg + '/'):
    os.makedirs('checkpoints/' + dq_mg + '/')
"""
Dueling DQN

CMS
python3 -m run --map CollectMineralShards --agent agents.dueling_DQN.Dueling_DQNMoveOnly (trained)
tensorboard --logdir=./tensorboard/DDQN_CMS

DR
python3 -m run --map DefeatRoaches --agent agents.dueling_DQN.Dueling_DQNMoveOnly (Currently training)
tensorboard --logdir=./tensorboard/DDQN_DR

DZB
python3 -m run --map DefeatZerglingsAndBanelings --agent agents.dueling_DQN.Dueling_DQNMoveOnly (trained)
tensorboard --logdir=./tensorboard/DDQN_DZB


watch model:
python -m run --map DefeatRoaches --agent agents.dueling_DQN.Dueling_DQNMoveOnly --ckpt_name=DDQN_DR --training=False

continue training model: 
python -m run --map DefeatRoaches --agent agents.dueling_DQN.Dueling_DQNMoveOnly --ckpt_name=DDQN_DR
"""

class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class Dueling_DQNMoveOnly(base_agent.BaseAgent):
    """A DQN that receives `player_relative` features and takes movements."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 discount_factor=FLAGS.discount_factor,
                 epsilon_max=FLAGS.epsilon_max,
                 epsilon_min=FLAGS.epsilon_min,
                 epsilon_decay_steps=FLAGS.epsilon_decay_steps,
                 train_frequency=FLAGS.train_frequency,
                 target_update_frequency=FLAGS.target_update_frequency,
                 max_memory=FLAGS.max_memory,
                 batch_size=FLAGS.batch_size,
                 training=FLAGS.training,
                 indicate_nonrandom_action=FLAGS.indicate_nonrandom_action,
                 save_dir="./checkpoints/" + dq_mg + "/",
                 ckpt_name=dq_mg,
                 summary_path="./tensorboard/" + dq_mg):
        """Initialize rewards/episodes/steps, build network."""
        super(Dueling_DQNMoveOnly, self).__init__()

        # saving and summary writing
        if FLAGS.save_dir:
            save_dir = FLAGS.save_dir
        if FLAGS.ckpt_name:
            ckpt_name = FLAGS.ckpt_name
        if FLAGS.summary_path:
            summary_path = FLAGS.summary_path

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # agent hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # other parameters
        self.training = training
        self.indicate_nonrandom_action = indicate_nonrandom_action

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.PlayerRelativeMovementCNN(
            spatial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        if self.training:
            self.target_net = nets.PlayerRelativeMovementCNN(
                spatial_dimensions=feature_screen_size,
                learning_rate=self.learning_rate,
                name="DQNTarget")

            # initialize Experience Replay memory buffer
            self.memory = Memory(max_memory)   # FIXME: save most recent actions
            self.batch_size = batch_size

        print("Done.")

        self.last_state = None
        self.last_action = None

        # initialize session
        config = tf.ConfigProto()                   # FIXME: make tensorflow use memory that it needs, instead of pre-allocating the memory
        config.gpu_options.allow_growth=True        # Actual FIXME: the training crashed while i was away, could this be the issue? Will test after the current training is done
        self.sess = tf.Session(config=config)
        #self.sess = tf.Session()

        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
            if self.training:
                self._update_target_network()
        else:
            self._tf_init_op()

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_state = None
            self.last_action = None

            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        screen = obs.observation.feature_screen.player_relative

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()
##########################################################################################################################################
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # array shape: (feature_screen_size, feature_screen_size)
            state = obs.observation.feature_screen.player_relative



            if self.training:
                # predict an action to take and take it
                x, y, action = self._epsilon_greedy_action_selection(state)

                # update online DQN FIXME: update
                if (self.steps % self.train_frequency == 0 and
                        len(self.memory) > self.batch_size):
                    self._train_network()

                # update network used to estimate TD targets FIXME: sync the two
                if self.steps % self.target_update_frequency == 0:
                    self._update_target_network()       # FIXME: make sure that the two NN is the same, purpose: unknown
                    print("Target network updated.")

                # add experience to memory
                if self.last_state is not None:
                    self.memory.add(
                        (self.last_state,
                         self.last_action,
                         obs.reward,
                         state))

                self.last_state = state
                self.last_action = np.ravel_multi_index(        # Get the xy coordinate of the map the agent clicked on in the last state
                    (x, y),
                    feature_screen_size)

            else:
                x, y, action = self._epsilon_greedy_action_selection(
                    state,
                    self.epsilon_min)

            if self.indicate_nonrandom_action and action == "nonrandom":
                # cosmetic difference between random and Q based actions
                return FUNCTIONS.Attack_screen("now", (x, y))
            else:
                return FUNCTIONS.Move_screen("now", (x, y))
        else:
            return FUNCTIONS.select_army("select")
########################################################################################################################
    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        states, actions, targets = self._get_batch()
        self.network.write_summary(
            self.sess, states, actions, targets, self.reward)
        print("Summary Written")

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQNTarget")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)

    def _epsilon_greedy_action_selection(self, state, epsilon=None):
        """Choose action from state with epsilon greedy strategy."""
        step = self.network.global_step.eval(session=self.sess)

        # epsilon is value for exploration
        # FIXME: understand how epsilon is implemented
        if epsilon is None:
            epsilon = max(
                self.epsilon_min,
                (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) *
                                     step / self.epsilon_decay_steps)))
########################################################################################################################
        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])

            return x, y, "random"

        else:
            inputs = np.expand_dims(state, 0)

            # Q values obtained from a frame/step
            # The Q values of all possible moves
            q_values = self.sess.run(
                self.network.flatten,
                feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values)

            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y, "nonrandom"
########################################################################################################################

    # FIXME: crucial part
    def _train_network(self):
        states, actions, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets)

    def _get_batch(self):
        """
        Notes for values below it.
        batch:  default: size 16, ie 16 different experiences/steps
                a list of experience/step. Search "add experience/step to memory" or look at the next few lines to know more
                about the structure of a experience/step

        states: default: (16, 84, 84).
                In this case there are 16 past experiences/stepa, each representing the feature_screen

        actions: default: size 16 array

        rewards: default: a size 16 array
                 The returned reward for this current experience/step for all 16 experience/step

        next_states: similar to states, but is a successor for each corresponding element in "states"

        :return:
        states:  described above
        actions: described above
        targets: size 16 list, used for calculating loss
        """

        batch = self.memory.sample(self.batch_size)                 # 16 different experiences
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # one-hot encode actions
        # actions: size (16, 7056) array
        # in the new actions, each element in each row represents the position on the feature screen. If the value is
        # 1, then it means the agent chose that position. This is then used for training.
        actions = np.eye(np.prod(feature_screen_size))[actions]

        # Use a CNN to predict the next_outputs
        # next_outputs: size (16, 7056)
        next_outputs = self.sess.run(
            self.target_net.output,
            feed_dict={self.target_net.inputs: next_states})

        # targets: size 16 list, used for calculating loss
        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets
