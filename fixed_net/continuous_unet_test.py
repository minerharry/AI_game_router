from tqdm import tqdm
from unet import UNet
from torch import Tensor, nn
import torch
import numpy as np

model = UNet(2,1);
print(model);

inp = torch.rand((2,2,17,17));
out = model(inp);
print(out);

epochs = 10;

num_data = 1000
size = (20,20);
center = [s/2 for s in size];
line_map = {};

for i in range(size[0]):
  for j in range(size[1]):
    hit_rects = [];
    line_end = (i+0.5,j+0.5);
    for t in range(size[0]):
      for l in range(size[1]):
        # from https://gist.github.com/ChickenProp/3194723
        p_1 = line_end[0]-center[0];
        p_2 = -p_1;
        p_3 = line_end[1]-center[1];
        p_4 = -p_3;
        p = [p_1,p_2,p_3,p_4];

        q_1 = center[0] - t;
        q_2 = t+1 - center[0];
        q_3 = center[1] - l;
        q_4 = l+1 - center[1];
        q = [q_1,q_2,q_3,q_4];
        miss = False;

        if t==l and i ==1 and j == -1:
          print(t,l);
          print(p,q);

        for qk,pk in zip(q,p):
          if (qk < 0 and pk == 0):
            miss = True;
            break;
        if t==l and i == 1 and j == -1:
          print(miss);
        if miss:
          continue;

        u_1 = -max([qk/pk for qk,pk in zip(q,p) if pk < 0])
        u_2 = -min([qk/pk for qk,pk in zip(q,p) if pk > 0]);
        # print(u_1,u_2);
        if u_1 >= u_2 and u_1 > 0 and u_2 <= 1:
          hit_rects.append((t,l));
        if t==l and i == 1 and j == -1:
          print(u_1,u_2);
        
    line_map[(i,j)] = hit_rects;

dataset = []
for i in tqdm(range(num_data)):
  in_im = np.random.default_rng().integers(0,2,size=size,dtype=np.uint8)
  out_im = np.ones(shape=size,dtype=np.float32);
  for x in range(size[0]):
    for y in range(size[1]):
      collisions = 0;
      for pixel in line_map[(x,y)]:
        if (in_im[pixel]):
          collisions += 1;
      out_im[x,y] = (1-collisions/len(line_map[(x,y)]));
  dataset.append([in_im,out_im]);

epochs = 10
learning_rate = 1e-2
batch_size = 64


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")