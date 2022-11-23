from __future__ import annotations
import copy
import heapq as hq
import itertools
from typing import Callable, DefaultDict, Generic, Iterable, Iterator, TypeVar

N = TypeVar('N');
T = TypeVar('T');
P = TypeVar('P')

class DStarQueue(Generic[T,P]):
    def __init__(self,start:list[tuple[P,T]]=[]):
        self._queue = start;
        hq.heapify(self._queue);

    def topKey(self):
        hq.heapify(self._queue)
        # print(self._queue)
        if len(self._queue) > 0:
            return self._queue[0][0]
        else:
            # print('empty queue!')
            return (float('inf'), float('inf'))

    def push(self,item:T,priority:P):
        hq.heappush(self._queue,(priority,item));

    def pop(self):
        return hq.heappop(self._queue);

    def __bool__(self):
        return len(self._queue) > 0;

    def __iter__(self)->Iterator[tuple[P,T]]:
        return iter(self._queue);

    def __contains__(self,item:T):
        return any([x for x in self if x[1] is item]);

    def pushOrUpdate(self,item:T,priority:P):
        for index,(_,i) in enumerate(self):
            if i == item:
                self._queue[index] = (priority,item);
                hq.heapify(self._queue);
                return;
        self.push(item,priority);

    def removeItem(self,item:T)->bool:
        for (p,i) in self:
            if i == item:
                self._queue.remove((p,i));
                return True;
        return False;

class DStarSearcher(Generic[N]):
    def __init__(self,heuristic:Callable[[N,N],float],start:N,goal:N,pred:Callable[[N],Iterable[tuple[N,float]]],succ:Callable[[N],Iterable[tuple[N,float]]],checkpoint:dict|None=None):

        self.start = start;
        self.goal = goal;
        self.h = heuristic;
        self.pred = pred;
        self.succ = succ;
        self.km = 0;
        if checkpoint is None:
            self.g = DefaultDict[N,float](lambda: float('inf'));
            self.rhs = DefaultDict[N,float](lambda: float('inf'));
            self.rhs[goal] = 0;
            self.U = DStarQueue[N,tuple[float,float]]();
            self.U.push(self.goal,(self.h(self.start,self.goal),0));
        else:
            self.load_checkpoint(checkpoint);

    def get_checkpoint_data(self):
        return {'g':dict(self.g),'rhs':dict(self.rhs),'U':self.U};

    def load_checkpoint(self,checkpoint):
        self.g = DefaultDict[N,float](lambda:float('inf'),checkpoint['g']);
        self.rhs = DefaultDict[N,float](lambda:float('inf'),checkpoint['rhs']);
        self.U = checkpoint['U'];

    def calculateKey(self,s:N): #somewhat equivalent to A*'s f_score
        t = min(self.g[s],self.rhs[s]) #the minimum of the two interpretations of the distance to the end
        return (self.km+self.h(self.start,s)+t,t) #(total path length, path to end [for tiebreaker])

    def updateVertex(self,u:N):
        if (self.g[u] != self.rhs[u]):
            self.U.pushOrUpdate(u,self.calculateKey(u));
        else:
            self.U.removeItem(u);

    def computeShortestPath(self):
        #while [start is not on the best path] or [start is locally inconsistent]
        while self.U.topKey() < self.calculateKey(self.start) or self.rhs[self.start] > self.g[self.start]:
            # print('path changed, exploring from pos',self.start);
            # print("Top key:",self.U.topKey(),"with start key",self.calculateKey(self.start));
            k_old,u = self.U.pop();
            k_new = self.calculateKey(u);
            # print(u)
            if k_old < k_new:
                # print('pushed - preds not checked')
                self.U.pushOrUpdate(u,k_new);
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u];
                self.U.removeItem(u);
                for s,c in self.pred(u):
                    self.rhs[s] = min(self.rhs[s],c+self.g[u]);
                    self.updateVertex(s);
            else:
                g_old = self.g[u];
                self.g[u] = float('inf');
                for s,c in itertools.chain(self.pred(u),[(u,0),]):
                    if self.rhs[s] == c + g_old and s != self.goal:
                        self.rhs[s] = min([c+self.g[sp] for sp,c in self.succ(s)]);
                    self.updateVertex(s);

        # print(self.g[self.start],self.rhs[self.start]);


    def search(self,
            move_func:Callable[[DStarSearcher],N], #callback for moving
            scan_func:Callable[[],list[tuple[N,N,float,float]]] #start,end,old,new
            ):
        self.computeShortestPath();
        last = self.start;
        while self.start != self.goal:
            next_node = move_func(self); #step forward
            changed_costs = scan_func();
            self.start = next_node;
            if any(changed_costs):
                self.km += self.h(last,self.start);
                last = self.start;
                for (u,v,c_old,c) in changed_costs:
                    if (u != self.goal):
                        if (c_old > c):
                            self.rhs[u] = min(self.rhs[u],c+self.g[v]);
                        elif self.rhs[u] == c_old + self.g[v]:
                            self.rhs[u] = min([cp+self.g[sp] for sp,cp in self.succ(u)]);
                    self.updateVertex(u);
                self.computeShortestPath();

#TODO: Consider implementing D* reset https://www.researchgate.net/publication/316945804_D_Lite_with_Reset_Improved_Version_of_D_Lite_for_Complex_Environment 

    def step_search(self,changed_costs:list[tuple[N,N,float,float]]=[]): #start,end,old,new
        if any(changed_costs):
            for (u,v,c_old,c) in changed_costs:
                if (u != self.goal):
                    if (c_old > c):
                        self.rhs[u] = min(self.rhs[u],c+self.g[v]);
                    elif self.rhs[u] == c_old + self.g[v]:
                        self.rhs[u] = min([cp+self.g[sp] for sp,cp in self.succ(u)]);
                self.updateVertex(u);
        self.computeShortestPath();

    def load_start(self,start:N):
        self.old_start = self.start;
        self.start = start;
        # self.km = self.h(self.old_start,self.start);

    def revert_start(self):
        self.start = self.old_start;
        # self.km = 0;

    def explore_from_position(self,pos:N): #note: if pos has already been explored with no updates, this should do little to nothing
        self.load_start(pos);
        self.computeShortestPath();
        self.revert_start();

    def return_shortest_path(self,start=None):
        if start:
            self.load_start(start);
            self.computeShortestPath();
        current = self.start;
        path = [current];
        while current != self.goal:
            
            current,_ = min([(s,c) for s,c in self.succ(current)],key=lambda e: self.g[e[0]] + e[1]);
            path.append(current);
        if start:
            self.revert_start();
        return path;        

    def update_costs(self,updates:list[tuple[N,N,float,float]]):
        for (u,v,c_old,c) in updates:
            if (u != self.goal):
                if (c_old > c):
                    self.rhs[u] = min(self.rhs[u],c+self.g[v]);
                elif self.rhs[u] == c_old + self.g[v]:
                    self.rhs[u] = min([cp+self.g[sp] for sp,cp in self.succ(u)]);
            self.updateVertex(u);
        self.computeShortestPath();




class LevelSearcher(Generic[N,T]):
    def __init__(self,start:N,is_goal:Callable[[N],bool],heuristic:Callable[[N],float],node_key:Callable[[N],T],succ:Callable[[N],Iterable[N]],cost:Callable[[N,N],float],frustration_multiplier:float,checkpoint:dict|None=None):
        self.start = start;
        self.goal = is_goal;

        self.c = cost;
        self.h = heuristic;

        self.succ = succ;     
        self.node_key = node_key;

        self.frustration_weight = frustration_multiplier;

        self.cameFrom = {};

        def sort_key(edge:tuple[N|None,N]):
            prev,node = edge;
            if (prev == None):
                raise Exception("attempting to sort initial edge, bad juju");
            return min(self.g[self.node_key(prev)].values()) + self.frustration[edge]*self.frustration_weight + self.c(prev,node) + self.h(node);
        
        self.sort_key = sort_key;
        
        if checkpoint is None:
            self.g = DefaultDict[T,dict[N|None,float]](lambda: {None:float('inf')});
            self.g[self.node_key(self.start)][None] = 0;
            self.frustration:dict[tuple[N|None,N],int] = DefaultDict(lambda:0);
            self.open_edges:list[tuple[N|None,N]] = [(None,start)]; #priority list of (prev,current) sorted by the sum of: a) the g score to the previous node, b) the estimated cost (self.c) from previous to current, and c) the heuristic from current
            self.completed_edges:list[tuple[N|None,N]] = [];
            self.complete_edge((None,start));
        else:
            self.load_checkpoint(checkpoint);
    
    def sorted_edges(self)->list[tuple[N|None,N]]:
        self.open_edges.sort(key=self.sort_key);
        return copy.deepcopy(self.open_edges);
        
    def complete_edge(self,edge:tuple[N|None,N]):
        if edge in self.open_edges:
            self.open_edges.remove(edge);
            self.completed_edges.append(edge);
            self.open_edges += [(edge[1],succ) for succ in self.succ(edge[1])];
        elif edge in self.completed_edges:
            print("WARNING: attempting to complete edge that has already been completed");
        else:
            print(f"WARNING: attempting to complete edge {edge} not in open edges FIX SOON");
            # raise Exception(f"Attempt to complete edge {edge} not in open edges or previously completed")

    def update_scores(self,scores:Iterable[tuple[N,float]]):
        for node,score in scores:
            self.g[self.node_key(node)][node] = score;

    def register_attempts(self,paths_attempted:Iterable[tuple[N|None,N]]):
        for path in paths_attempted:
            self.frustration[path] += 1;

            

    def get_checkpoint_data(self):
        return {'g':dict(self.g),'open':self.open_edges,'complete':self.completed_edges,'frustration':dict(self.frustration)};

    def load_checkpoint(self,checkpoint:dict):
        self.g = DefaultDict[T,dict[N|None,float]](lambda: {None:float('inf')},checkpoint['g']);
        self.open_edges = checkpoint['open'];
        self.completed_edges = checkpoint['complete'] if 'complete' in checkpoint else [];
        self.frustration = DefaultDict[tuple[N|None,N],int](lambda:0,checkpoint['frustration'] if 'frustration' in checkpoint else {});
        
        


        

