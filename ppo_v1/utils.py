import os, pickle
import matplotlib.pyplot as plt

def _ensure_optimizer_built(optimizer, variables):
    if hasattr(optimizer, "built") and not optimizer.built:
        optimizer.build(variables)


def save_training_state(base_actor_path, agent, actor_opt, critic_opt, game_seeds):
    _ensure_optimizer_built(actor_opt, agent.actor.cnn.trainable_variables)
    _ensure_optimizer_built(critic_opt, agent.critic.cnn.trainable_variables)

    # Try to extract optimizer weights in a robust way (different TF/Keras versions expose different APIs)
    def _get_opt_weights(opt):
        try:
            return opt.get_weights()
        except AttributeError:
            # Fall back to reading variables as numpy arrays
            try:
                return [v.numpy() for v in opt.variables]
            except Exception:
                return None

    training_data = {
        "rolling_mean_history": agent.rolling_mean_history,
        "rolling_mean_store": agent.rolling_mean_history,
        "reward_history": agent.reward_history,
        "step_history": agent.step_history,
        "total_steps": agent.total_steps,
        "total_updates": agent.total_updates,
        "agent_reward_history": agent.reward_history,
        "game_seeds": game_seeds,
        "actor_optimizer_weights": _get_opt_weights(actor_opt),
        "critic_optimizer_weights": _get_opt_weights(critic_opt),
        "actor_optimizer_iterations": int(actor_opt.iterations.numpy()) if hasattr(actor_opt, 'iterations') else None,
        "critic_optimizer_iterations": int(critic_opt.iterations.numpy()) if hasattr(critic_opt, 'iterations') else None,
    }

    file_path = base_actor_path.replace('actor_model', 'training_data.pkl')
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def save(actorPath, criticPath, CHECKPOINTS, agent, CHECKPOINTFREQ, cycle, rolling, act_opt, critic_opt, game_seeds, final, asBest = False):
    if asBest:
        saveBest(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds)
    if final:
        saveFinal(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds, rolling, CHECKPOINTS)
    if agent.total_steps % CHECKPOINTFREQ == 0 and agent.total_steps > 0 and CHECKPOINTS:
        saveCheckpoint(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds, cycle)
    
def saveBest(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds):
    best_actorPath = actorPath.replace('/check/', '/best/')
    best_criticPath = criticPath.replace('/check/', '/best/')
    agent.saveModels(actor_path=best_actorPath, critic_path=best_criticPath, checkpoint=True)
    save_training_state(best_actorPath, agent, act_opt, critic_opt, game_seeds)
    print(f"Best Models saved to {best_actorPath} and {best_criticPath} ")
    
def saveCheckpoint(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds, cycle):
    checkpoint_actorPath = actorPath.replace('/check/', f'/{cycle}/')
    checkpoint_criticPath = criticPath.replace('/check/', f'/{cycle}/')
    agent.saveModels(actor_path=checkpoint_actorPath, critic_path=checkpoint_criticPath, checkpoint=True)
    save_training_state(checkpoint_actorPath, agent, act_opt, critic_opt, game_seeds)
    print(f"Checkpoint Models saved to {checkpoint_actorPath} and {checkpoint_criticPath} ")
    
def saveFinal(actorPath, criticPath, agent, act_opt, critic_opt, game_seeds, rolling, CHECKPOINTS):
    if CHECKPOINTS:
        replace = '/x/check/'
    else:
        replace = '/x/'
    actorPath = actorPath.replace(replace, f'/{rolling:.0f}/')
    criticPath = criticPath.replace(replace, f'/{rolling:.0f}/')
        
    agent.saveModels(actor_path=actorPath, critic_path=criticPath, temp=f'{agent.rolling_mean_history[-1]:.0f}', checkpoint=False, saveCheckpoints=CHECKPOINTS)

def plotGraph(BORDERTHICKNESS, LINETHICKNESS, LATEX, STYLE, DIAGRAMWIDTH, SOLUTIONTHRESHOLD, FIGURENAME, agent):
    
    
    plt.rcParams['grid.linewidth'] = BORDERTHICKNESS
    plt.rcParams['xtick.major.width'] = BORDERTHICKNESS
    plt.rcParams['ytick.major.width'] = BORDERTHICKNESS
    plt.rcParams['axes.linewidth'] = BORDERTHICKNESS
    plt.rcParams['lines.linewidth'] = LINETHICKNESS

    width = 20
    height = 10
    
    if LATEX: # https://duetosymmetry.com/code/latex-mpl-fig-tips/
        plt.rcParams.update({'text.usetex':True})
        try:
            plt.style.use(STYLE)
        except (OSError, FileNotFoundError):
            print(f"Warning: Style file '{STYLE}' not found. Using default matplotlib style.")
            plt.style.use('default')
        pt = 1./72.27
        golden = (1 + 5 ** 0.5) / 2
        width = DIAGRAMWIDTH * pt
        height = width/golden

    plt.figure(figsize = (width,height))
    plt.plot(agent.step_history, agent.reward_history, label = r"PPO $\mu_{D_k}$", alpha = 0.9)
    plt.plot(agent.step_history, agent.rolling_mean_history, label = r"$\text{SMA}_{50}$", linewidth=1)
    plt.hlines(y=SOLUTIONTHRESHOLD, xmin=0, xmax= agent.total_steps, colors='r', linestyles='--', linewidth=1, alpha= 0.9)
    plt.xlabel(f"Step total")
    plt.ylabel("Episodic Reward")
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()
    plt.savefig(FIGURENAME)
    
def loadModel(agent, loadPathActor, loadPathCritic, act_opt, critic_opt, actor, critic):
    agent.loadModels(actor_path=loadPathActor, critic_path=loadPathCritic)
        
    filePath = loadPathActor.replace('actor_model', 'training_data.pkl')
    
    try:
        with open(filePath, "rb") as f:
            data = pickle.load(f)

        rolling_hist = data.get("rolling_mean_history") or data.get("rolling_mean_store")
        if rolling_hist is not None:
            agent.rolling_mean_history = rolling_hist

        agent.reward_history = data.get("reward_history", agent.reward_history)
        agent.step_history = data.get("step_history", agent.step_history)
        agent.total_updates = data.get("total_updates", agent.total_updates)
        agent.total_steps = data.get(
            "total_steps",
            agent.step_history[-1] if agent.step_history else agent.total_steps
        )

        loaded_game_seeds = data.get("game_seeds")
        if loaded_game_seeds is not None and len(loaded_game_seeds) > 0:
            game_seeds = loaded_game_seeds

        actor_opt_weights = data.get("actor_optimizer_weights")
        if actor_opt_weights:
            _ensure_optimizer_built(act_opt, actor.cnn.trainable_variables)
            act_opt.set_weights(actor_opt_weights)

        actor_opt_iterations = data.get("actor_optimizer_iterations")
        if actor_opt_iterations is not None:
            act_opt.iterations.assign(actor_opt_iterations)

        critic_opt_weights = data.get("critic_optimizer_weights")
        if critic_opt_weights:
            _ensure_optimizer_built(critic_opt, critic.cnn.trainable_variables)
            critic_opt.set_weights(critic_opt_weights)

        critic_opt_iterations = data.get("critic_optimizer_iterations")
        if critic_opt_iterations is not None:
            critic_opt.iterations.assign(critic_opt_iterations)

        if not agent.reward_history:
            agent.reward_history = [0]
        if not agent.step_history:
            agent.step_history = [agent.total_steps]
        if not agent.rolling_mean_history:
            agent.rolling_mean_history = [0]

        print(f"Loaded training state from {filePath}")
        
        return game_seeds
    except FileNotFoundError:
        print("No stored data found")