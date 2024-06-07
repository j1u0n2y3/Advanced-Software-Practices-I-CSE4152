import torch
import numpy as np

class ClassificationNetworkColors(torch.nn.Module):
    def __init__(self):
        super(ClassificationNetworkColors, self).__init__()

        self.classes = [[-1., 0., 0.],  # left
                        [-1., 0.5, 0.], # left and accelerate
                        [-1., 0., 0.8], # left and brake
                        [1., 0., 0.],   # right
                        [1., 0.5, 0.],  # right and accelerate
                        [1., 0., 0.8],  # right and brake
                        [0., 0., 0.],   # no input
                        [0., 0.5, 0.],  # accelerate
                        [0., 0., 0.8]]  # brake

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=16, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 64, kernel_size=8, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, len(self.classes))
        )

    def forward(self, observation):
        observation = observation.permute(0, 3, 1, 2)
        conv_out = self.conv_layers(observation)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        return self.linear_layers(conv_out)

    def actions_to_classes(self, actions):
        classes = []
        for action in actions:
            action = action.tolist()
            if action not in self.classes:
                distances = [np.linalg.norm(np.array(action) - np.array(cls)) for cls in self.classes]
                action = self.classes[np.argmin(distances)]
            classes.append(torch.tensor([self.classes.index(action)], dtype=torch.long))
        return classes

    def scores_to_action(self, scores):
        _, predicted = torch.max(scores, 1)
        return self.classes[predicted.item()]
