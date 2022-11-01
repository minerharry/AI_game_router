import time
import ray
import argparse

@ray.remote(resources={"Display":1})
def pwned():
    print("pwning...");
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
    print("parsing inputs stuff");
    time.sleep(30);
    parser = argparse.ArgumentParser(prog = "multi node display resource tester");
    parser.add_argument("head_node_ip");
    parser.add_argument("-n","--no_ray",dest='no_ray',action='store_const',const=True,default=False);
    args = parser.parse_args();
    ip = args.head_node_ip;
    import logging
    logging.info("logging???");

    no_ray = parser.parse_args().no_ray;
    if not no_ray:
        import ray
        print("starting ray: connecting to remote ip",ip);
        ray.init(log_to_driver=False);
        import logging
        logging.info("HELP I CAN'T SEE");
        print("awaiting resource availability");
        r = pwned.remote();
        ray.get(r);
    else:
        print("not using ray, running pygame");
        pwned._function();