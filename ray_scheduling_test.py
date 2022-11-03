import time
import ray
from ray.util.multiprocessing import Pool
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import sys

ray.init(address=sys.argv[1] or "auto");

print("waiting for display node...");
most_display = None;
while most_display is None:
    nodes = ray.nodes();
    max_display = 0;
    for n in nodes:
        if "Display" in n["Resources"]:
            d = n["Resources"]["Display"];
            if d > max_display:
                most_display = n["NodeID"];
                max_display = d
    time.sleep(5);
print("display node obtained:",most_display);

@ray.remote(scheduling_strategy=NodeAffinitySchedulingStrategy(most_display,soft=True))
class contextActor:
    def show_context(self):
        context = ray.get_runtime_context();
        c = context.get()
        c["resources"] = context.get_assigned_resources();
        print(c);
        time.sleep(4);
        return c;

print("Cluster total resources:",ray.cluster_resources());
print("Cluster resource availability:",ray.available_resources());
print("Cluster nodes:",ray.nodes());
refs = [contextActor.remote() for _ in range(20)];
[c.show_context.remote() for c in refs];
ray.get(refs);

    

