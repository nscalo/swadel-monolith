import os
import numpy as np
import torch
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from dataset import mnist_data
from utils import Logger
from torch.autograd.variable import Variable
from tqdm import tqdm
from vision import images_to_vectors, vectors_to_images, noise, ones_target, zeros_target
from DiscrimatorNet import DiscrimatorNet
from GeneratorNet import GeneratorNet
from optimize import train_discriminator, train_generator, optimizer_funcs
from torch.nn import BCELoss

if __name__ == "__main__":

    # Load data
    data = mnist_data()

    data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

    num_batches = len(data_loader)

    logger = Logger(model_name="VGAN1", data_name="MNIST")

    num_epochs = 60

    loss = BCELoss()

    generator = GeneratorNet(100, 784)
    discriminator = DiscrimatorNet(784, 1)

    discrimator_optim, generator_optim = optimizer_funcs(discriminator, generator)

    for epoch in tqdm(range(num_epochs)):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            N = real_batch.size(0)

            real_data = Variable(images_to_vectors(real_batch))
            fake_data = generator(noise(N)).detach()
            d_error, d_pred_real, d_pred_fake = train_discriminator(discrimator_optim, discriminator, loss, real_data, fake_data)

            fake_data = generator(noise(N))
            g_error = train_generator(generator_optim, discriminator, loss, fake_data)
            logger.log(d_error, g_error, epoch, n_batch, num_batches)

            if (n_batch) % 200 == 0:
                num_test_samples = 16
                test_noise = noise(num_test_samples)
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data

                logger.log_images(
                    test_images, num_test_samples, epoch, n_batch, num_batches
                )
                logger.display_status(
                    epoch, num_epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
        if epoch % 10 == 0:
            logger.save_models(generator, discriminator, epoch)