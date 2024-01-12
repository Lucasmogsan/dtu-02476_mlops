import pstats
import os

script_path = os.path.dirname(os.path.abspath(__file__))

rows_shown = 20
file_name = "outputs/profiling/profiling_output.txt"
file_path = os.path.join(script_path, file_name)
sorting_order = "tottime"  # 'cumulative', 'tottime', 'time', 'ncalls', 'pcalls' (see https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats)

p = pstats.Stats(file_name)
p.sort_stats(sorting_order).print_stats(rows_shown)
