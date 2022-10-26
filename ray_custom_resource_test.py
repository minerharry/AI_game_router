import time
import ray

@ray.remote(resources={"Display":1})
def pwned():
    import pygame as pg
    pg.init();
    pg.display.set_caption("I AM HACKING YOUR COMPUTER MWAHAHAHA");
    pg.display.set_mode((500,40));

    s = pg.display.get_surface();

    f = pg.font.Font('arial.ttf',20);

    text = f.render("you just got pwned from a google cloud server!!!!",True,(255,0,0),(0,0,0));

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
    ray.init();
    r = pwned.remote();
    ray.get(r);