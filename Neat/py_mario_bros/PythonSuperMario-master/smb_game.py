from baseGame import RunGame
from abc import abstractmethod





class SMB1Game(RunGame):
    
    def __init__(self,runnerConfig,kwargs):
        self.steps = 0;
        self.runConfig = runnerConfig;
        self.game = kwargs["game"];
        self.game.load_segment(runnerConfig.training_datum);
        


    @abstractmethod
    def getOutputData(self):
        return self.game.get_game_data();

    @abstractmethod
    def processInput(self, inputs):
        output = [key > 0 for key in inputs];
        named_actions = zip(['action','jump','left','right','down'],output);
        self.game.tick_inputs(named_actions);
        

    @abstractmethod
    def renderInput(self,inputs):
        pass;

    def close(self):
        #does nothing unless game needs it to
        return;