""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torch.nn import ModuleList


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, expansion=64,depth=4,bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        current_size = expansion;
        self.inc = DoubleConv(n_channels, current_size);

        self.down = ModuleList();
        for _ in range(depth-1):
            self.down.append(Down(current_size, current_size*2));
            current_size = current_size * 2;
        
        factor = 2 if bilinear else 1
        self.down.append(Down(current_size, current_size * 2 // factor)); #last one's funky
        current_size = current_size * 2 // factor;

        self.up = ModuleList();
        for _ in range(depth):
            self.up.append(Up(current_size, current_size//2, bilinear));
            current_size = current_size // 2;

        self.outc = OutConv(current_size, n_classes)
        # self.register_parameter(name="up_convs",param=self.up);
        # self.register_parameter(name="down_convs",param=self.down);

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        for i,d in enumerate(self.down):
            self.down[i] = d.to(*args,**kwargs);
        for i,u in enumerate(self.up):
            self.up[i] = u.to(*args,**kwargs);
        return self

    def forward(self, x):

        x_layers = [];
        x_layers.append(self.inc(x));
        for d in self.down:
            x_layers.append(d(x_layers[-1]));

        x = x_layers.pop();
        for u in self.up:
            x = u(x,x_layers.pop());

        logits = self.outc(x)
        return logits
