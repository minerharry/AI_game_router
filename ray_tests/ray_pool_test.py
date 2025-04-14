import time
import ray
from ray.util.multiprocessing import Pool 

ray.init();
num_procs = 3;
p = Pool(num_procs);

lengths = [1,15,1,1,1,1,1,1]; ## put long task on first actor; blocks other actors from receiving subsequent tasks

print("ideal time:",sum(lengths)/num_procs);

start = time.time();

class ContextArr:
    def __init__(self,data):
        self.data = data;


def wait_time(i,arr):
    t = arr[i];
    print(f"waiting {t} seconds...",f"@{time.time()-start}")
    time.sleep(t);
    print(f"done!",f"@{time.time()-start}");
    time.sleep(0.5);
res = p.imap_unordered(wait_time,lengths,chunksize=1);
list(res); #clear queue for iterators

end = time.time();

print("actual time:",end-start);


lengths = [1,2,1,10,3,1,4,10,1,3,2,10]; ## put long tasks all on one actor due to cyclic nature

lengths.index(1);