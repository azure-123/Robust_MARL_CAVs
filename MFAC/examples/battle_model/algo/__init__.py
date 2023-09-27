from . import mfac
from . import q_learning
from . import ac

MFQ = q_learning.MFQ
DQN = q_learning.DQN
MFAC = mfac.MFAC
AC = ac.AC

def spawn_ai(algo_name, env,  human_name, max_steps):
    if algo_name == 'mfq':
        model = MFQ(env, max_steps, memory_size=80000)
    elif algo_name == 'dqn':
        model = DQN( env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC( env)
    elif algo_name =="ac":
        model = AC( env)
    return model
