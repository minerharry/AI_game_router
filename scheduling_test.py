import ray
from ray.experimental.state.api import list_nodes
from ray.util.multiprocessing.pool import Pool
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

ray.init();

nodes = list_nodes();
special_node = [n for n in nodes if "Special" in n["resources_total"]][0];
special_id = special_node["node_id"];

task_pool = Pool(ray_remote_args={"scheduling_strategy":NodeAffinitySchedulingStrategy(special_id,True)})


