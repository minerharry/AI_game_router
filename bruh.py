

from typing import Any, Callable


class MetaRunner(type):
    def __getattr__(cls,key):
        print("boo!")
        print(cls);
        if hasattr(cls,key):
            return getattr(super(),key);
        else:
            print(key);
    pass;


class Runner(metaclass=MetaRunner):
    def __init__(self,data:str):
        self.data = data;

    @classmethod
    def run(cls,data:str,target:Callable[[str],Any]):
        runner = Runner(data);
        print(runner);
        # return runner.run(target);

    def run(self,target:Callable[[str],Any]):
        print("running!")
        return target(self.data);


class FooType(type):
    def _foo_func(cls):
        return 'foo!'

    def _bar_func(cls):
        return 'bar!'

    def __getattr__(cls, key):
        if key == 'Foo':
            return cls._foo_func()
        elif key == 'Bar':
            return cls._bar_func()
        raise AttributeError(key)

    def __str__(cls):
        return 'custom str for %s' % (cls.__name__,)

class MyClass(metaclass=FooType):
    pass



import code
code.interact(local=locals());