from neat.reporting import BaseReporter
import os


def FitnessReporter(BaseReporter):

    def __init__(self,gameName,run_name):
        os.makedirs(f"memories\\{gameName}\\{run_name}_history",exist_ok=True);

        self.gameName = gameName;
        self.run_name = run_name;