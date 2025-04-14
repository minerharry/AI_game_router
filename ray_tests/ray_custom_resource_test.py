import time
import ray
import argparse
import ray
import sys

@ray.remote(resources={"Display":1})
def pwned():
    print("pwning...");
    import pygame as pg
    pg.init();
    pg.display.set_caption("I AM HACKING YOUR COMPUTER MWAHAHAHA");
    pg.display.set_mode((500,40));

    s = pg.display.get_surface();

    f = pg.font.Font('arial.ttf',20);

    text = f.render("you just got pwned from the longleaf server!!!!",True,(255,0,0),(0,0,0));

    print(text.get_size());

    speed = 50;#units/sec
    
    reset = 700;
    maxSlice = text.get_width();

    last = time.time();
    while True:
        now = time.time();
        slicepos = (now-last)*speed
        if (slicepos > reset):
            last = now;
            slicepos = 0;
            print("reset");
        slicepos = min(slicepos,maxSlice);
        pg.event.pump();
        s.fill((0,0,0));
        s.blit(text,(0,0),pg.rect.Rect(0,0,slicepos,text.get_height()));
        pg.display.flip();


if __name__ == "__main__":
    print("parsing inputs stuff");
    ip = sys.argv[1] or "auto"
    print("starting ray: connecting to remote ip",ip);

    ray.init(address=ip,log_to_driver=False);
    print("awaiting resource availability");
    r = pwned.remote();
    ray.get(r);
