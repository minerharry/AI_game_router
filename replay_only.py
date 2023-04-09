


from training_data import TrainingDataManager


replay_runConfig = copy.deepcopy(runConfig)
replay_runConfig.logging = False;
replay_runConfig.training_data = Traini ngDataManager("smb1Py","replay");
replay_runConfig.reporters = [];
replay_game = copy.deepcopy(game);
replay_game.initInputs.update({"window_title":"Best Genomes - Instant Replay"})
replay = ReplayRenderer.remote(replay_game,checkpoint_run_name,replay_runConfig)