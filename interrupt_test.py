

from interrupt import DelayedKeyboardInterrupt


interruped = False;

while not interruped:
    with DelayedKeyboardInterrupt():
        for i in range(200):
            print(i);