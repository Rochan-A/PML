import tensorflow as tf
import time

import os
import os.path as osp
import joblib
import numpy as np

from tensorboardX import SummaryWriter

    # Initialize Trainer
    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        config=config,
        writer=writer,
        no_test_flag=args.no_test_flag,
        only_test_flag=args.only_test_flag
    )


class Trainer(object):
    """
    Training script for Learning to Adapt

    Args:
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        initial_random_sampled (bool) : Whether or not to collect random samples in the first iteration
        dynamics_model_max_epochs (int): Number of epochs to train the dynamics model
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """

    def __init__(
        self,
        env,
        env_flag,
        sampler,
        sample_processor,
        policy,
        dynamics_model,
        n_itr,
        writer,
        start_itr=0,
        initial_random_samples=True,
        num_rollouts=10,
        dynamics_model_max_epochs=200,
        test_max_epochs=200,
        sess=None,
        context=False,
        num_test=4,
        test_range=[[1.0, 2.0], [3.0, 4.0], [16.0, 17.0], [18.0, 19.0]],
        total_test=20,
        no_test_flag=False,
        only_test_flag=False,
        use_cem=False,
        horizon=0,
        test_num_rollouts=10,
        test_n_parallel=2,
        history_length=10,
        state_diff=False,
        mcl_dynamics_model=None,
        cadm_n_epochs=0,
        mcl_cadm=False,
    ):

        # Environment Attirubtes
        self.env = env
        self.env_flag = env_flag

        # Sampler Attributes
        self.sampler = sampler
        self.sample_processor = sample_processor

        # Dynamics Model Attributes
        self.dynamics_model = dynamics_model

        # Policy Attributes
        self.policy = policy
        self.use_cem = use_cem
        self.horizon = horizon

        # Algorithm Attributes
        self.context = context

        # CaDM Attributes
        self.context = context
        self.history_length = history_length
        self.state_diff = state_diff

        # MCL Attributes
        self.mcl_cadm = mcl_cadm

        # Training Attributes
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_rollouts = num_rollouts
        self.dynamics_model_max_epochs = dynamics_model_max_epochs
        self.test_max_epochs = test_max_epochs
        self.initial_random_samples = initial_random_samples

        # Testing Attributes
        self.no_test = no_test_flag
        self.only_test = only_test_flag
        self.total_test = total_test
        self.num_test = num_test
        self.test_range = test_range
        self.writer = writer
        self.test_num_rollouts = test_num_rollouts
        self.test_n_parallel = test_n_parallel

        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
        self.sess = sess

    def train(self):
        """
        Collects data and trains the dynamics model
        """
        f_test_list = []
        for i in range(0, self.num_test):
            file_name = "%s/test_c%d.txt" % (logger.get_dir(), i)
            f_test = open(file_name, "w+")
            f_test_list.append(f_test)

        file_name = "%s/test_tot.txt" % (logger.get_dir())
        f_test_tot = open(file_name, "w+")

        file_name = "%s/train.txt" % (logger.get_dir())
        f_train = open(file_name, "w+")

        itr_times = []
        t0 = time.time()

        test_env_list = []

        train_env = env_cls()
        train_env.seed(0)
        train_env = normalize(train_env)
        for i in range(0, self.num_test):
            test_env = env_cls(self.test_range[i][0], self.test_range[i][1])
            test_env.seed(0)
            test_env = normalize(test_env)
            vec_test_env = ParallelEnvExecutor(
                test_env,
                self.test_n_parallel,
                self.test_num_rollouts,
                self.test_max_epochs,
                True,
            )
            test_env_list.append(vec_test_env)

        if len(train_env.action_space.shape) == 0:
            act_dim = train_env.action_space.n
            discrete = True
        else:
            act_dim = train_env.action_space.shape[0]
            discrete = False

        if args.use_reward_model:
            reward_config["model_config"]["load_model"] = True
            reward_model = RewardModel(reward_config=reward_config)
        else:
            reward_model = None

        # initial MPC controller
        mpc_controller = MPC(mpc_config=mpc_config, reward_model=reward_model)

        if args.train_reward_model:
            reward_model = RewardModel(reward_config=reward_config)
        """NN pretrain"""
        pretrain_episodes = 40
        for epi in range(pretrain_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs_next, reward, done, state_next = env.step(action)
                model.add_data_point([0, obs, action, obs_next - obs])
                if args.train_reward_model: 
                    reward_model.add_data_point([0, obs_next, action, [reward]])
                obs = obs_next
        # training the model
        model.fit()
        if args.train_reward_model:
            print("********** fitting reward model **********")
            reward_model.fit()

        """testing the model with MPC while training """
        test_episode = 3
        test_epoch = 20
        
        for ep in range(test_epoch):
            print('epoch: ', ep)
            
            for epi in range(test_episode):
                obs = env.reset()
                acc_reward, done = 0, False
                mpc_controller.reset()
                i = 0
                while not done:
                    i+= 1
                    if args.render:
                        env.render()
                    action = np.array([mpc_controller.act(model=model, state=obs)])
                    obs_next, reward, done, state_next = env.step(action)

                    model.add_data_point([0, obs, action, obs_next - obs])
                    if args.train_reward_model: 
                        reward_model.add_data_point([0, obs_next, action, [reward]])

                    obs = obs_next
                    acc_reward += reward

                print('step: ', i, 'acc_reward: ', acc_reward)
                env.close()

                if done:
                    print('******************')
                    print('acc_reward', acc_reward)

            print('********** fitting the dynamic model **********')
            model.fit()
            if args.train_reward_model:
                print("********** fitting reward model **********")
                reward_model.fit()


    def get_itr_snapshot(self, itr):
        """
        Gets the current policy, env, and dynamics model for storage
        """
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
            dynamics_model=self.dynamics_model,
        )

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
