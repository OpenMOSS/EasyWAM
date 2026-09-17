"""RoboDojo client loop for a remote EasyWAM XPolicyLab server."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
            if index + 1 < len(actions):
                model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        if not indices:
            break
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices))
        chunks = model_client.call(func_name="get_action_batch", obs=indices)
        if len(chunks) != len(indices) or not chunks or not chunks[0]:
            raise ValueError("Policy returned an invalid action batch.")
        if any(len(chunk) != len(chunks[0]) for chunk in chunks):
            raise ValueError("Policy returned unequal action chunk lengths.")
        for step in range(len(chunks[0])):
            TASK_ENV.take_action_batch([chunk[step] for chunk in chunks], indices)
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [offset for offset, index in enumerate(indices) if index in running]
            indices = [indices[offset] for offset in active]
            chunks = [chunks[offset] for offset in active]
            if not indices:
                break
            if step + 1 < len(chunks[0]):
                model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices))
