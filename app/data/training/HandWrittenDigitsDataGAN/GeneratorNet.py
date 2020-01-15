from torch import nn

class GeneratorNet(nn.Module):

    def __init__(self, n_features, n_out):
        super(GeneratorNet, self).__init__()
        self.n_features = n_features
        self.n_out = n_out

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.out = nn.Sequential(
            nn.Linear(1024, self.n_out),
            nn.Tanh()
        )

    def forward(self, _input):
        _input = self.hidden0(_input)
        _input = self.hidden1(_input)
        _input = self.hidden2(_input)
        _input = self.out(_input)

        return _input

    