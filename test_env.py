from envs import ContexualEnv, DummyContextualEnv
import sys
import yaml, time, torch
from easydict import EasyDict
import random
from envs.term_rew import cartpole_upright_reward, cartpole_upright_term, cartpole_swingup_rew, cartpole_swingup_term

if __name__=='__main__':

    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    env_fam = ContexualEnv(config)

    print('#'*80)
    print('sampled training contexts')
    for _ in range(10):
        env, context = env_fam.reset(train=True)
        print(context)

    print('#'*80)
    print('sampled testing contexts')
    for _ in range(10):
        env, context = env_fam.reset(train=False)
        print(context)

    cartpole_swingup_rew_ = cartpole_swingup_rew(env.l)
    s = env.reset()
    done = False
    while not done:
        a = random.random()*2 - 1
        s_, r, done, _ = env.step([a])
        env.render('human')
        time.sleep(0.2)
        print(r, cartpole_swingup_rew_(torch.tensor([[a]]), torch.tensor([s_])), done, cartpole_swingup_term(torch.tensor([[a]]), torch.tensor([s_])), s)
        s = s_