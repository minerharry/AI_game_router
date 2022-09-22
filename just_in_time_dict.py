from __future__ import annotations
from collections import UserDict
from typing import Any, Callable, DefaultDict, Mapping, TypeVar


_KT = TypeVar('_KT')
_VT = TypeVar('_VT');
Delegate = Callable[[],_VT];

class JITDict(UserDict[_KT,_VT|Delegate]):
    def __init__(self,*args,delegates=Mapping[_KT,Delegate],**kwargs):
        self.func = set();
        super().__init__(*args,**kwargs);
        self.updateDelegates(delegates);
    def updateDelegates(self,other=(),**kwds):
        if isinstance(other, Mapping):
            for key in other:
                self.setdelegate(key,other[key]);
        elif hasattr(other, "keys"):
            for key in other.keys():
                self.setdelegate(key, other[key]);
        else:
            for key, value in other:
                self.setdelegate(key, value);
        for key, value in kwds.items():
            self.setdelegate(key, value);
    def setdelegate(self,__key:_KT,delegate:Delegate):
        self.func.add(__key);
        return super().__setitem__(__key,delegate);
    def __setitem__(self, __key: _KT, item: _VT) -> None:
        return super().__setitem__(__key, item)
    def __getitem__(self, __key: _KT) -> Any:
        # print(f"item get: {__key}!");
        if __key in self.func:
            res = super().__getitem__(__key).__call__();
            # print(f"item result obtained: {res}")
            super().__setitem__(__key,res);
            self.func.remove(__key);
            return res;
        return super().__getitem__(__key);


if __name__ == "__main__":
    def calc_1():
        print("calculated 1");
        return 1;

    def calc_2():
        print("calculated 2");
        return 2;
    
    def calc_3():
        print("calculated 3");
        return 3;

    
