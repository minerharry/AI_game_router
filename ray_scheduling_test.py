import time
from numpy import disp
import ray
from ray.util.multiprocessing import Pool
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group, placement_group_table
import sys
import os

ray.init(address=sys.argv[1] or "auto");

print("waiting for display node...");
num_display = 0
while num_display < 2:
    r = ray.cluster_resources();
    if "Display" in r:
        num_display = r["Display"]
    time.sleep(5);
print("display node obtained, display cores available:",num_display);

basic_cores = ray.cluster_resources()["CPU"]-num_display-2; #two extra cores for whatever

cpu_bundles = [{"CPU":1} for _ in range(int(basic_cores))];
display_bundles = [{"Display":0.01,"CPU":1} for _ in range(int(num_display))];

total_bundles = cpu_bundles + display_bundles
group = placement_group(total_bundles,strategy="SPREAD");
ray.get(group.ready());
print(placement_group_table(group));
st = PlacementGroupSchedulingStrategy(group);

@ray.remote(scheduling_strategy=st)
class cActor():
    def show_context():
        context = ray.get_runtime_context();
        c = context.get()
        c["resources"] = context.get_assigned_resources();
        print(c);
        print(os.environ["SDL_VIDEODRIVER"] if "SDL_VIDEODRIVER" in os.environ else None);
        time.sleep(4);
        return c['node_id'];

print("Cluster total resources:",ray.cluster_resources());
print("Cluster resource availability:",ray.available_resources());
print("Cluster nodes:",ray.nodes());
refs = [cActor.remote() for c in range(len(total_bundles))];
t = [c.show_context.remote() for c in refs];

ids = ray.get(t);

print([f"{id}: {ids.count(id)}" for id in set(ids)]);

    

