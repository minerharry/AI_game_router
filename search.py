from __future__ import annotations
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
    def __init__(self,heuristic:Callable[[N,N],float],start:N,goal:N,pred:Callable[[N],Iterable[tuple[N,float]]],succ:Callable[[N],Iterable[tuple[N,float]]]):
        self.g = DefaultDict[N,float](lambda: float('inf'));
        self.rhs = DefaultDict[N,float](lambda: float('inf'));
        self.start = start;
        self.goal = goal;
        self.rhs[goal] = 0;
        self.h = heuristic;
        self.pred = pred;
        self.succ = succ;
        self.km = 0;
        self.U = DStarQueue[N,tuple[float,float]]();
        self.U.push(self.goal,(self.h(self.start,self.goal),0));


    def calculateKey(self,s:N):
        t = min(self.g[s],self.rhs[s])
        return (t+self.h(self.start,s)+self.km,t)

    def updateVertex(self,u:N):
        if (self.g[u] != self.rhs[u]):
            self.U.pushOrUpdate(u,self.calculateKey(u));
        else:
            self.U.removeItem(u);

    def computeShortestPath(self):
        while self.U.topKey() < self.calculateKey(self.start) or self.rhs[self.start] > self.g[self.start]:
            k_old,u = self.U.pop();
            k_new = self.calculateKey(u);
            if k_old < k_new:
                self.U.pushOrUpdate(u,k_new);
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u];
                self.U.removeItem(u);
                for s,c in self.pred(u):
                    if s != self.goal:
                        self.rhs[s] = min(self.rhs[s],c+self.g[u]);
                    self.updateVertex(s);
            else:
                g_old = self.g[u];
                self.g[u] = float('inf');
                for s,c in itertools.chain(self.pred(u),[(u,0),]):
                    if self.rhs[s] == c + g_old and s != self.goal:
                        self.rhs[s] = min([c+self.g[sp] for sp,c in self.succ(s)]);
                    self.updateVertex(s);


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

    #allows control by an external loop by iterating over the function to prompt recalculation
    #DOES NOT MOVE
    def search_iter(self,
            scan_func:Callable[[],list[tuple[N,N,float,float]]]
            ): #start,end,old,new
        self.computeShortestPath();
        yield self;
        while True:
            changed_costs = scan_func();
            if any(changed_costs):
                for (u,v,c_old,c) in changed_costs:
                    if (u != self.goal):
                        if (c_old > c):
                            self.rhs[u] = min(self.rhs[u],c+self.g[v]);
                        elif self.rhs[u] == c_old + self.g[v]:
                            self.rhs[u] = min([cp+self.g[sp] for sp,cp in self.succ(u)]);
                    self.updateVertex(u);
                self.computeShortestPath();
            yield self;         



class LevelSearcher(Generic[N,T]):
    def __init__(self,start:N,is_goal:Callable[[N],bool],heuristic:Callable[[N],float],node_key:Callable[[N],T],succ:Callable[[N],Iterable[N]],cost:Callable[[N,N],float]):
        self.start = start;
        self.goal = is_goal;

        self.open_edges:list[N] = []; #priority list sorted by the sum of: a) the g-score of the cost between N and T [as it gets updated] and b) the heuristic from T
        
        self.c = cost;
        self.g = DefaultDict[T,dict[N|None,float]](lambda: {None:float('inf')});
        self.h = heuristic;

        self.succ = succ;     
        self.node_key = node_key;   

        self.g[self.node_key(self.start)][None] = 0;
        
        self.cameFrom = {};

        def sort_key(node:N):
            return min(self.g[self.node_key(node)].values()) + self.h(node);
        
        self.sort_key = sort_key;

    def sorted_edges(self)->list[N]:
        self.open_edges.sort(key=self.sort_key);
        return self.open_edges;
        

    def complete_edge(self,edge:N):
        self.open_edges.remove(edge);
        self.open_edges += [succ for succ in self.succ(edge)];

    def update_scores(self,scores:Iterable[tuple[N,float]]):
        for node,score in scores:
            self.g[self.node_key(node)][node] = score;
        


        

