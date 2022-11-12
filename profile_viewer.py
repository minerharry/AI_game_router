import pstats
from pstats import SortKey
p = pstats.Stats('profile_logs/fido/t3_p0_render_profile.stats');
p.strip_dirs();
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats();

p.print_callers("gameReporting.py:41")