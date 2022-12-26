from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState


state = SegmentState(None,None,file_path='levels/testing/test2.lvl');
game = tools.Control();
state_dict = {c.LEVEL: Segment()}
game.setup_states(state_dict, c.LEVEL)
game.state.startup(0,{c.LEVEL_NUM:1},initial_state=state);
game.main(do_fps=True);