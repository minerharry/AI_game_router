

def f_1():
    di = {'0':0,'1':1,'2':2,'3':3};
    f_2(di);
    print(di);
    f_2(di);
    print(di);


def f_2(kwargs):
    kwargs['0'] += 1;
    kwargs['1'] = kwargs['2'];
    

if __name__ == "__main__":
    f_1();