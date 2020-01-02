import matplotlib.pyplot as plt

def get_file_content():
    losses = []
    with open('log/losses.txt','r') as file:
        lines = file.readlines()
        #losses += [float(line)]
        for l in lines:
            losses += [float(l)]
    return losses

plt.ion()
import time
plt.figure()
while True:
    losses = get_file_content()
    plt.clf()
    plt.semilogy(losses)
    plt.show()
    plt.pause(3)
