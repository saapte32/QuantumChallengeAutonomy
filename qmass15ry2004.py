import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pennylane as qml
from torchvision.utils import save_image
import math

batch_size = 8   #change your batch size
lrG = 0.3  # Learning rate for the generator
lrD = 0.01  # Learning rate for the discriminator
num_iter=500
image_size = 64
workers=0

ngpu=1
dataroot = "bdd105"

# Enable CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataset = dset.ImageFolder(root=dataroot,
                        transform=transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)


# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.imshow(real_batch)

# Quantum variables
n_qubits = 9  # Total number of qubits / N
n_a_qubits = 1  # Number of ancillary qubits / N_A
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 48 # Number of subgenerators for the patch method / N_G
image_size = 64



# Quantum simulator
dev = qml.device("lightning.qubit", wires=n_qubits)


@qml.qnode(dev, diff_method="parameter-shift")
def quantum_circuit(noise, weights):

    weights = weights.reshape(q_depth, n_qubits)

    # Initialise latent vectors
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)

    # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits):
            qml.RY(weights[i][y], wires=y)

        # Control Z gates
        for y in range(n_qubits - 1):
            qml.CZ(wires=[y, y + 1])

    return qml.probs(wires=list(range(n_qubits)))


# For further info on how the non-linear transform is implemented in Pennylane
# https://discuss.pennylane.ai/t/ancillary-subsystem-measurement-then-trace-out/1532
def partial_measure(noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(noise, weights)
    probsgiven0 = probs[: (2 ** (n_qubits - n_a_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand(q_depth * n_qubits), requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators

    def forward(self, x):
        # Size of each sub-generator output
        patch_size = 2 ** (n_qubits - n_a_qubits)

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, patch_size).to(device)
            for elem in x:
                q_out = partial_measure(elem, params).float().unsqueeze(0)
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images

generator = PatchQuantumGenerator(n_generators).to(device)

print(generator)

# Classical Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
           nn.Linear(3*image_size * image_size, 64,device=device),
            nn.ReLU(),
            nn.Linear(64, 64,device=device),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(64, 16,device=device),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(16, 1,device=device),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.reshape(int(b_size), -1)
        return self.model(x)


# Initialize discriminator
discriminator = Discriminator().to(device)

print(discriminator)

# Iteration counter
counter = 0
# Binary cross entropy
criterion = nn.BCELoss()
# Collect images for plotting later
results = []




# Optimisers
optD = optim.SGD(discriminator.parameters(), lr=lrD)
optG = optim.SGD(generator.parameters(), lr=lrG)

real_labels = torch.full((batch_size,), 1.0, dtype=torch.float, device=device)
fake_labels = torch.full((batch_size,), 0.0, dtype=torch.float, device=device)



fixed_noise = torch.rand(batch_size, n_qubits, device=device) * math.pi 

save_path='losses/quantum_mod.pt'

def save_checkpoint(gen,dis, optg,optd, save_path):
    torch.save({
        'gen_state_dict': gen.state_dict(),
        'dis_state_dict': dis.state_dict(),
        'optG_state_dict': optg.state_dict(),
        'optD_state_dict': optd.state_dict()

    }, save_path)
def load_checkpoint(gen,dis, optg,optd, load_path):
    checkpoint = torch.load(load_path)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    dis.load_state_dict(checkpoint['dis_state_dict'])
    optg.load_state_dict(checkpoint['optG_state_dict'])
    optd.load_state_dict(checkpoint['optD_state_dict'])


    return gen,dis, optg, optd




G_losses = []
D_losses = []

while True:
    for i, (data, _) in enumerate(dataloader):


            # Data for training the discriminator
            data = data.reshape(-1,image_size * image_size*3)

            real_data = data.to(device)
            b_size = real_data.size(0)
            
            
            real_labels = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
            fake_labels = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
        


            # Training the discriminator

            noise = torch.rand(b_size, n_qubits, device=device) * math.pi / 2
            fake_data = generator(noise)
    
            discriminator.zero_grad()
            outD_real = discriminator(real_data).view(-1)

            errD_real = criterion(outD_real, real_labels)
            
            # Noise following a uniform distribution in range [0,pi/2)
            outD_fake = discriminator(fake_data.detach()).view(-1)
            errD_fake = criterion(outD_fake, fake_labels)
            # Propagate gradients
            errD_real.backward()
            errD_fake.backward()

            errD = errD_real + errD_fake
            optD.step()



            # Training the generator
            generator.zero_grad()
        
            outD_fake = discriminator(fake_data).view(-1)
        
            errG = criterion(outD_fake, real_labels)
            errG.backward()
            optG.step()

            counter += 1
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Show loss values
            if counter % 1 == 0:
                print(f'Iteration: {counter}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')
                test_images = generator(fixed_noise).view(batch_size,3,image_size,image_size).cpu().detach()

                # Save images every 10 iterations
                if counter % 10 == 0:
                    results.append(test_images)
                    for j in range(len(results)):
                        save_image(results[j], '../quantumchallengeautonomy/ryimages5/gen_img_ry'+str(j)+'.png')

            if counter == num_iter:
                break
    if counter == num_iter:
        break




save_checkpoint(generator,discriminator, optG,optD, save_path)

load_checkpoint(generator,discriminator, optG,optD, save_path)

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()




for k in range(len(test_images)) :

      #fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True, figsize=(2,2), facecolor='white')
      fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=True, figsize=(2,2), facecolor='white')
      #save_image(test_images[k], '../quantumchallengeautonomy/ryimages/gen_img_ry'+str(k)+'.png')

      axs.matshow(np.squeeze(test_images[k].permute(1,2,0)))
