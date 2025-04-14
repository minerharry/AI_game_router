from contextlib import contextmanager
import functools
import time
import ray
from ray.util.multiprocessing import Pool 

ray.init();
num_procs = 3;
p = Pool(num_procs);

lengths = [0,1,2,3,4,5,6,7]; ## put long task on first actor; blocks other actors from receiving subsequent tasks

print("ideal time:",sum(lengths)/num_procs);

start = time.time();

class ContextArr:
    def __init__(self,data):
        self.data = data;

    def __getitem__(self,index:int):
        return self.data[index];

    # def __call__(self,tempData):
    #     return self.manager(tempData)

    # def manager(self,tempData):
    #     pass;

    @contextmanager
    def temp_change_data(self,tempData):
        self.temp = self.data;
        self.data = tempData;
        yield tempData;
        self.data = self.temp;

    def __iter__(self):
        return iter(self.data);

    def __len__(self):
        return len(self.data);


def wait_time(i,arr=[]):
    t = arr[i];
    print(f"waiting {t} seconds...",f"@{time.time()-start}")
    time.sleep(t);
    print(f"done!",f"@{time.time()-start}");
    time.sleep(0.5);
    return i;

array = ContextArr(lengths)
res = p.imap_unordered(functools.partial(wait_time,arr=lengths),array,chunksize=1);
print(list(res)); #clear queue for iterators

end = time.time();

print("actual time:",end-start);

with array.temp_change_data([5,4,3,2,1]):
    res = p.imap_unordered(functools.partial(wait_time,arr=lengths),array,chunksize=1);
    print(array.data);
    print(list(res)); #clear queue for iterators

print(array.data)

# lengths = [1,2,1,10,3,1,4,10,1,3,2,10]; ## put long tasks all on one actor due to cyclic nature

# lengths.index(1);

input();