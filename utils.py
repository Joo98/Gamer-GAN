import torch
from torchvision.utils import save_image
import numpy as np

def train(dataloader, generator, discriminator, optimzer_G, optimizer_D, loss, epochs, sample_interval, device):
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.ones((imgs.size(0), 1), dtype=torch.float).to(device)
            fake = torch.ones((imgs.size(0), 1), dtype=torch.float).to(device)
            
            #valid = Tensor(imgs.size(0), 1).fill_(1.0), 
            #fake = Tensor(imgs.size(0), 1).fill_(0.0),

            # Configure input
            real_imgs = imgs.to(device)
            
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = (np.random.normal(0, 1, (imgs.shape[0], generator.latent_dim)))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
