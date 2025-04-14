import pygame as pg
import sys
from pathlib import Path

if __name__=='__main__':
    print(str(Path(sys.path[0]).parent.parent.parent.parent));
    sys.path.append(str(Path(sys.path[0]).parent.parent.parent.parent))
    from source.main import main
    main()
    pg.quit()