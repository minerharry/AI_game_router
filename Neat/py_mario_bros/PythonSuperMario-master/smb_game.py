from baseGame import RunGame




class SMB1Game(RunGame):
    def __init__(self,runnerConfig,kwargs):
        self.steps = 0;
        self.runConfig = runnerConfig;
        self.game = tools.Control()
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                  c.LOAD_SCREEN: load_screen.LoadScreen(),
                  c.LEVEL: level.Level(),
                  c.GAME_OVER: load_screen.GameOver(),
                  c.TIME_OUT: load_screen.TimeOut()}
        self.game.setup_states(state_dict, c.MAIN_MENU)
        
        


    @abstractmethod
    def getOutputData(self):
        #return dict of all data available from game, sans 'steps'
        pass;

    @abstractmethod
    def processInput(self, inputs):
        output = [key > 0 for key in inputs];
        named_actions = zip(['action','jump','left','right','down'],output);
        

    @abstractmethod
    def renderInput(self,inputs):
        pass;

    def close(self):
        #does nothing unless game needs it to
        return;