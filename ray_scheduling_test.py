import time
import ray
from ray.util.multiprocessing import Pool
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group
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

group = placement_group(cpu_bundles + display_bundles,strategy="SPREAD");
st = PlacementGroupSchedulingStrategy(group);

@ray.remote(scheduling_strategy=st)
class contextActor:
    def show_context(self):
        context = ray.get_runtime_context();
        c = context.get()
        c["resources"] = context.get_assigned_resources();
        print(c);
        print(os.environ["SDL_VIDEODRIVER"])
        time.sleep(4);
        return c;

print("Cluster total resources:",ray.cluster_resources());
print("Cluster resource availability:",ray.available_resources());
print("Cluster nodes:",ray.nodes());
refs = [contextActor.remote() for _ in range(20)];
t = [c.show_context.remote() for c in refs];

ray.get(t);

    

