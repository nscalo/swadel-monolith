from torch import nn, optim
from vision import ones_target, zeros_target

def optimizer_funcs(discriminator, generator):
    discrimator_optim = optim.Adam(discriminator.parameters(), lr=0.0002)
    generator_optim = optim.Adam(generator.parameters(), lr=0.0002)

    return discrimator_optim, generator_optim

def train_discriminator(optimizer, discriminator, loss, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, discriminator, loss, fake_data):
    N = fake_data.size(0)

    optimizer.zero_grad()

    prediction = discriminator(fake_data)

    error = loss(prediction, ones_target(N))
    error.backward()

    optimizer.step()

    return error

