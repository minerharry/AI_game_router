import pygame as pg
pg.init();
SCREEN = pg.display.set_mode((400,400))


center_left = 50;
center_top = 50;
rect_width = 10;
view_distance = 2;
rects = [[pg.Rect(center_left + i * rect_width, center_top + j * rect_width,rect_width,rect_width) for j in range(-view_distance,view_distance+1)] for i in range(-view_distance,view_distance+1)]
print(rects)

spriteRects = [pg.Rect(4*rect_width,13/2*rect_width,7/2*rect_width/2,2*rect_width)];
print([[1 if rect.collidelist(spriteRects) >= 0 else 0 for rect in row] for row in rects]);

checker_colors = [(20,20,20),(230,230,230)];

for i in range(len(rects)):
    row = rects[i]
    for j in range(len(row)):
        rect = row[j]
        pg.draw.rect(SCREEN,(i*256/len(rects),128,j*256/len(row)),rect);

for rect in spriteRects:
    pg.draw.rect(SCREEN,(255,0,0),rect)



pg.display.update();
done = False;
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True