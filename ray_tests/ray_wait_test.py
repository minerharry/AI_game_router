import time
import ray
from ray.util.multiprocessing import Pool 

ray.init();
num_procs = 3;
p = Pool(num_procs);

lengths = [1,15,1,1,1,1,1,1]; ## put long task on first actor; blocks other actors from receiving subsequent tasks

print("ideal time:",sum(lengths)/num_procs);

start = time.time();

@ray.remote
def wait_time(t):
    print(f"waiting {t} seconds...",f"@{time.time()-start}")
    time.sleep(t);
    print(f"done!",f"@{time.time()-start}");
    time.sleep(0.5);

refs = [wait_time.remote(l) for l in lengths];
ref_unready = refs;
for _ in lengths:
    ready, ref_unready = ray.wait(ref_unready);
    unready = ref_unready
    ready = [refs.index(r) for r in ready];
    ready = [(r,lengths[r]) for r in ready];
    unready = [refs.index(r) for r in unready];
    unready = [(r,lengths[r]) for r in unready];

    print(f"@{time.time()-start}",ready,unready);
    time.sleep(0.5);


end = time.time();

print("actual time:",end-start);


lengths = [1,2,1,10,3,1,4,10,1,3,2,10]; ## put long tasks all on one actor due to cyclic nature

lengths.index(1);