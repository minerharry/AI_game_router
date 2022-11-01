import time
import ray
from ray.util.multiprocessing import Pool
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


ray.init();
logger = logging.getLogger(__name__)

print("waiting for display node...");
logger.info("waiting for display node...");
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

@ray.remote(scheduling_strategy=NodeAffinitySchedulingStrategy(most_display,soft=True))
def show_context():
    context = ray.get_runtime_context();
    c = context.get()
    c["resources"] = context.get_assigned_resources();
    print(c);
    return c;


refs = [show_context.remote() for _ in range(20)];
print(ray.get(refs));

    

