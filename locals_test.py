

a = [];
b = 2;

print("locals:",locals());
print("globals:",globals());

def hello():
    print("locals:",locals());
    print("globals:",globals());
    global b
    print("locals:",locals());
    print("globals:",globals());
    b = 4
    print("locals:",locals());
    print("globals:",globals());
    global c
    print("locals:",locals());
    print("globals:",globals());
    c = 16
    print("locals:",locals());
    print("globals:",globals());


hello();
print("locals:",locals());
print("globals:",globals());
print(c);