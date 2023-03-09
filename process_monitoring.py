from gameReporting import GameReporter


class ProcessMonitor(GameReporter):
    def __init__(self):
        self.active_processes = [];
        