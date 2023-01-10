
import itertools
import random
from typing import Iterable, Literal

__all__ = ["simplify_rectangles"];

pt = tuple[int,int];

dirs = [0,1,2,3]; #left, top, right, bottom
offsets:dict[int,pt] = {0:(-1,0),1:(0,-1),2:(1,0),3:(0,1)};
dirnames:list[str] = ["left","top","right","bottom"];

def offsetPoints(points:Iterable[pt],offset:pt):
    for p in points:
        yield (p[0]+offset[0],p[1]+offset[1]);

class rect:
    def __init__(self,topleft:pt,size:pt):
        self.tl = topleft;
        self.size = size;

    @property
    def left(self):
        return self.tl[0];

    @property
    def top(self):
        return self.tl[1];

    @property
    def right(self):
        return self.tl[0] + self.size[0] - 1;

    @property
    def bottom(self):
        return self.tl[1] + self.size[1] - 1;
        
    @property
    def width(self):
        return self.size[0];

    @property
    def height(self):
        return self.size[1];

    def leftRow(self):
        for y in range(self.top,self.bottom+1):
            yield (self.left,y);
    
    def rightRow(self):
        for y in range(self.top,self.bottom+1):
            yield (self.right,y);

    def topRow(self):
        for x in range(self.left,self.right+1):
            yield (x,self.top);

    def bottomRow(self):
        for x in range(self.left,self.right+1):
            yield (x,self.bottom);

    def rectSide(self,dir:int):
        match dir:
            case 0:
                return self.leftRow();
            case 1:
                return self.topRow();
            case 2:
                return self.rightRow();
            case 3:
                return self.bottomRow();
            case _:
                raise Exception(f"illegal direction index: {dir}");

    def shiftDims(self,dir:int):
        match dir:
            case 0:
                self.tl = (self.left-1,self.top);
                self.size = (self.size[0]+1,self.size[1]);
            case 1:
                self.tl = (self.left,self.top-1);
                self.size = (self.size[0],self.size[1]+1);
            case 2:
                self.size = (self.size[0]+1,self.size[1]);
            case 3:
                self.size = (self.size[0],self.size[1]+1);
    
    def __iter__(self):
        return iter(itertools.product(range(self.left,self.right+1),range(self.top,self.bottom+1)));

    def __repr__(self):
        return str(self.tl) + " size:" + str(self.size);

class Continue(Exception): pass;

def simplify_randomFloodfill(points:Iterable[pt],numAttempts=50):
    bestRects = None;
    points = list(points);
    for _ in range(numAttempts):
        remainingPoints = list(points);

        rects:list[rect] = []
        
        while len(remainingPoints) > 0 and (bestRects is None or bestRects[1] > len(rects)):
            start = random.sample(remainingPoints,1)[0];
            remainingPoints.remove(start);
            currentRect = rect(start,(1,1));
            # print("new rect:",currentRect);
            while True:
                try:
                    for d in dirs:
                        pushRects = set(offsetPoints(currentRect.rectSide(d),offsets[d]));
                        if all(p in remainingPoints for p in pushRects): #all rects in offset are available
                            [remainingPoints.remove(r) for r in pushRects];
                            # print("shifting",dirnames[d],"rect:",currentRect);
                            currentRect.shiftDims(d);
                            # print(currentRect);
                            # assert all([r in points for r in currentRect])
                            raise Continue;
                except Continue:
                    continue;
                break;
            rects.append(currentRect);
            # assert all([any([p in r for r in rects]) for p in points if p not in remainingPoints]);
        # print("remaining:",remainingPoints);
        if len(remainingPoints) > 0:
            continue;
        
        if bestRects is None or bestRects[1] > len(rects):
            # print("new best:",len(rects),"with remaining:",remainingPoints);
            bestRects = (rects,len(rects));
    if bestRects is not None:
        return bestRects[0];
    else:
        raise Exception();
                    

def simplify_sweep(points:Iterable[pt]):
    raise NotImplementedError();

def simplify_rectangles(points:Iterable[pt],type:Literal["randomFloodfill","sweep"]="randomFloodfill",**kwargs):
    match type:
        case "randomFloodfill":
            return simplify_randomFloodfill(points,**kwargs);
        case "sweep":
            return simplify_sweep(points,**kwargs);
        case _:
            raise KeyError(type);

if __name__ == "__main__":
    import numpy as np
    squares = np.random.random((400,30)) > 0.7 #fewer squares than open spaces;
    print(squares.transpose().astype(int))
    p = np.where(squares == True);
    # print(p);
    coords = list(zip(*p));
    print(list(coords));
    rects = simplify_randomFloodfill(coords);
    print(list(coords),rects);
    if rects is not None:
        sq2 = np.zeros(squares.shape,dtype=int);
        for i,r in enumerate(rects):
            for p in r:
                try:
                    sq2[p] = i+1;
                except:
                    print(r);
                    raise Exception();
        print(len(rects));
        np.set_printoptions(linewidth=np.inf)
        print(sq2.transpose());
        s = ((sq2 > 0) == (squares));
        print(s.transpose());
        print(np.where(s != True));

