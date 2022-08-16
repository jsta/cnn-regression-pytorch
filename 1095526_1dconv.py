import pandas as pd
from sklearn.model_selection import train_test_split


import torch
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import L1Loss
from torch.nn import Flatten
from torch.optim import Adam
from torch.nn import MaxPool1d
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset

# install pytorch's ignite and then import R2 score package
# !pip install pytorch-ignite
# from ignite.contrib.metrics.regression.r2_score import R2Score

dataset = pd.read_csv("/content/drive/My Drive/Dataset_Assignment/housing.csv")
dataset = dataset.dropna()
dataset.head(10)

# plot each feature of the dataset on separate sub-plots
print("Plot for each feature of the dataset:")
dataset.plot(subplots=True, grid=True)

Y = dataset["median_house_value"]
X = dataset.loc[:, "longitude":"median_income"]
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=2003
)
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()
x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()


class CnnRegressor(torch.nn.Module):
    # defined the initialization method
    def __init__(self, batch_size, inputs, outputs):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()

        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs

        self.input_layer = Conv1d(inputs, batch_size, 1, stride=1)

        self.max_pooling_layer = MaxPool1d(1)

        self.conv_layer1 = Conv1d(batch_size, 128, 1, stride=3)
        self.conv_layer2 = Conv1d(128, 256, 1, stride=3)
        self.conv_layer3 = Conv1d(256, 512, 1, stride=3)

        self.flatten_layer = Flatten()

        self.linear_layer = Linear(512, 128)

        self.output_layer = Linear(128, outputs)

    def feed(self, input):
        input = input.reshape((self.batch_size, self.inputs, 1))
        output = relu(self.input_layer(input))
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer3(output))

        output = self.flatten_layer(output)
        output = relu(self.linear_layer(output))
        output = self.output_layer(output)

        return output


batch_size = 100
model = CnnRegressor(batch_size, X.shape[1], 1)

model.cuda()


def model_loss(model, dataset, train=False, optimizer=None):
    # first calculated for the batches and at the end get the average
    performance = L1Loss()
    # score_metric = R2Score()

    avg_loss = 0
    avg_score = 0
    count = 0

    for input, output in iter(dataset):        
        predictions = model.feed(input)
        
        loss = performance(predictions, output)

        # compute the R2 score
        # score_metric.update([predictions, output])
        # score = score_metric.compute()

        if train:
            # clear the errors
            optimizer.zero_grad()

            # compute the gradients for optimizer
            loss.backward()

            # use optimizer in order to update parameters
            # of the model based on gradients
            optimizer.step()
        
        avg_loss += loss.item()
        # avg_score += score
        count += 1

    return avg_loss / count, avg_score / count


epochs = 100

# define the performance measure and optimizer
# optimizer = SGD( model.parameters(), lr= 1e-5)
optimizer = Adam(model.parameters(), lr=0.007)

# to process with GPU, training set is converted into torch variable
inputs = torch.from_numpy(x_train_np).cuda().float()
outputs = torch.from_numpy(y_train_np.reshape(y_train_np.shape[0], 1)).cuda().float()

# create the DataLoader instance to work with batches
tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

# loop for number of epochs and calculate average loss
for epoch in range(epochs):    
    avg_loss, avg_r2_score = model_loss(model, loader, train=True, optimizer=optimizer)
    print(
        "Epoch "
        + str(epoch + 1)
        + ":\n\tLoss = "
        + str(avg_loss)
        + "\n\tR^2 Score = "
        + str(avg_r2_score)
    )

torch.save(
    model.state_dict(),
    "/content/drive/My Drive/Dataset_Assignment/1095526_1dconv_reg.h",
)

# to process with GPU, testing set is converted into torch variable
inputs = torch.from_numpy(x_test_np).cuda().float()
outputs = torch.from_numpy(y_test_np.reshape(y_test_np.shape[0], 1)).cuda().float()

# create the DataLoader instance to work with batches
tensor = TensorDataset(inputs, outputs)
loader = DataLoader(tensor, batch_size, shuffle=True, drop_last=True)

# output of the performance of the model
avg_loss, avg_r2_score = model_loss(model, loader)
print("The model's L1 loss is: " + str(avg_loss))
print("The model's R^2 score is: " + str(avg_r2_score))
