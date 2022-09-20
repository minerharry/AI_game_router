import pstats
from pstats import SortKey
p = pstats.Stats('smb1Py_profile.stats');
p.strip_dirs();
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(20);
# p.print_callees(100,"getOutputData");
p.print_callees("segment.py:1063");


# p.print_callees("draw");

# p.print_callees("update_player_position");
# p.print_callees("draw");