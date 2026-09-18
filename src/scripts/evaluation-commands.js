const evaluationCommands = {
  libero: [
    'python experiments/libero/run_libero_manager.py \\',
    '  task=libero_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  'libero-plus': [
    'python experiments/libero_plus/run_libero_plus_manager.py \\',
    '  task=libero_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robotwin: [
    'python experiments/robotwin/run_robotwin_manager.py \\',
    '  task=robotwin_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robodojo: [
    'python experiments/robodojo/run_robodojo_manager.py \\',
    '  task=robodojo_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt>',
  ],
  robocasa: [
    'python experiments/robocasa/run_robocasa_manager.py \\',
    '  task=robocasa_easywam_mot_wan22 \\',
    '  ckpt=<path/to/checkpoint.pt> \\',
    '  EVALUATION.dataset_stats_path=<path/to/dataset_stats.json>',
  ],
};
