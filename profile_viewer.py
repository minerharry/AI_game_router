import pstats
from pstats import SortKey
p = pstats.Stats('profile_logs/t1_p0_render_profile.stats');
p.strip_dirs();
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(50);