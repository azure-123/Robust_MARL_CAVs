import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    """
        height: int, height of agent body
        width:  int, width of agent body
        speed:  float, maximum speed, i.e. the radius of move circle of the agent
        hp:     float, maximum health point of the agent
        view_range: gw.CircleRange or gw.SectorRange

        damage: float, attack damage
        step_recover: float, step recover of health point (can be negative)
        kill_supply: float, the hp gain when kill this type of agents

        step_reward: float, reward get in every step
        kill_reward: float, reward gain when kill this type of agent
        dead_penalty: float, reward get when dead
        attack_penalty: float, reward get when perform an attack (this is used to make agents do not attack blank grid)
    """

    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 10, 'speed': 2,
            'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(2),
            'damage': 2, 'step_recover': 0.1, 'kill_reward': 5, 'dead_penalty': -0.1, 'attack_penalty': -0.3 # 攻击空单元
        })

    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 2, 'speed': 2.5,
            'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(0),
            'damage': 0, 'step_recover': 0, 'kill_reward': 100, 'dead_penalty': -0.5 # 死掉惩罚
        })

    predator_group  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(prey_group, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg
