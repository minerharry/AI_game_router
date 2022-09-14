


import functools
import dill

t = True;


class tester:
    def testing_thingy(self,a,b:dict,c,reload=None):
        global t
        if reload:
            a,b,c = dill.loads(reload);
        self.c = c;
        print(a,b,c);

        def dummy_func(x):
            return x+self.c;
        if t:
            b['function'] = dummy_func;
            t = False;

        a = (b['function'](a));
        print(a);

        pickler = (a,b,c);
        return dill.dumps(pickler);

if __name__ == "__main__":
    a = 10;
    b = {};
    c = 30;
    test = tester();
    pick = test.testing_thingy(a,b,c);
    newargs = dill.loads(pick);
    test.testing_thingy(newargs[0],newargs[1],20);

