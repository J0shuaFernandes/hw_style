import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from lenet50 import LeNet

import os
import joblib

if __name__ == '__main__':
  bs = 32
  epochs = 100  
  LEARNING_RATE = 0.001
  network_name = 'lenet50_sc'

  train_loader = DataLoader(joblib.load('dataset/train.pickle'), batch_size=bs)
  test_loader = DataLoader(joblib.load('dataset/test.pickle'), batch_size=bs)

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"

  print(f"Using {device}")
  net = LeNet().to(device)

  # initialise loss funtion + optimiser
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

  for i in range(epochs):
    print(f"Epoch {i+1}") 
    for x, y in train_loader:
      x, y = x.to(device), y.to(device)

      # calculate loss
      prediction = net(x)
      loss = loss_fn(prediction, y)

      # backpropagate error and update weights
      optimiser.zero_grad()
      loss.backward()
      optimiser.step()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            predicted = net(x)
            #print(output)
            for idx, i in enumerate(predicted):
                #print(torch.argmax(i), y[idx])
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    print("Accuracy: {} Loss: {}".format(round(correct/total, 3), loss.item()))
    
    print("---------------------------")

  torch.save(net.state_dict(), "{}.pth".format(network_name))
  print("Finished training")