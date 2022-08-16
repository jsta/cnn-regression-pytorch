import pandas as pd
from sklearn.model_selection import train_test_split

from torch.optim import Adam
from torch.nn import L1Loss

# install pytorch's ignite and then import R2 score package
# !pip install pytorch-ignite
# from ignite.contrib.metrics.regression.r2_score import R2Score

dataset = pd.read_csv("/content/drive/My Drive/Dataset_Assignment/housing.csv")
dataset = dataset.dropna()
dataset.head(10)

# plot each feature of the dataset on separate sub-plots
print("Plot for each feature of the dataset:")
dataset.plot(subplots=True, grid=True)

# + id="5UtwA5P4QEbz" colab_type="code" colab={}
# we have to predict the value of median house value so selected that column in Y
Y = dataset["median_house_value"]

# the othe remaining columns are selected in X
X = dataset.loc[:, "longitude":"median_income"]

# + id="SszHXOI3waDC" colab_type="code" colab={}
# split the dataset by maintaining the ratio of training to testing as 70:30 with random state as 2003
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=2003
)

# convert the values of training set to the numpy array
x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()

# convert the values of testing set to the numpy array
x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

# + id="XthOjeOyaRWy" colab_type="code" colab={}
import torch

# import 1D convolutional layer
from torch.nn import Conv1d

# import max pooling layer
from torch.nn import MaxPool1d

# import the flatten layer
from torch.nn import Flatten

# import linear layer
from torch.nn import Linear

# import activation function (ReLU)
from torch.nn.functional import relu

# import libraries required for working with dataset from pytorch
from torch.utils.data import DataLoader, TensorDataset


# + id="Kt2ntfQDaORo" colab_type="code" colab={}
# defined model named as CnnRegressor and
# this model should be the subclass of torch.nn.Module
class CnnRegressor(torch.nn.Module):
    # defined the initialization method
    def __init__(self, batch_size, inputs, outputs):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()
        # store the parameters
        self.batch_size = batch_size
        self.inputs = inputs
        self.outputs = outputs
        # define the input layer
        self.input_layer = Conv1d(inputs, batch_size, 1, stride=1)

        # define max pooling layer
        self.max_pooling_layer = MaxPool1d(1)

        # define other convolutional layers
        self.conv_layer1 = Conv1d(batch_size, 128, 1, stride=3)
        self.conv_layer2 = Conv1d(128, 256, 1, stride=3)
        self.conv_layer3 = Conv1d(256, 512, 1, stride=3)

        # define the flatten layer
        self.flatten_layer = Flatten()

        # define the linear layer
        self.linear_layer = Linear(512, 128)

        # define the output layer
        self.output_layer = Linear(128, outputs)

    # define the method to feed the inputs to the model
    def feed(self, input):
        # input is reshaped to the 1D array and fed into the input layer
        input = input.reshape((self.batch_size, self.inputs, 1))

        # ReLU is applied on the output of input layer
        output = relu(self.input_layer(input))

        # max pooling is applied and then Convolutions are done with ReLU
        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = relu(self.conv_layer3(output))

        # flatten layer is applied
        output = self.flatten_layer(output)

        # linear layer and ReLu is applied
        output = relu(self.linear_layer(output))

        # finally, output layer is applied
        output = self.output_layer(output)
        return output


# + id="vUJzFuFBaYVZ" colab_type="code" outputId="9b77ea77-a2f6-4f3e-9f54-6166878f12ab" colab={"base_uri": "https://localhost:8080/", "height": 185}
# define the batch size
batch_size = 100
model = CnnRegressor(batch_size, X.shape[1], 1)

# we are using GPU so we have to set the model for that
model.cuda()


# + id="_i5I8CQXacaG" colab_type="code" colab={}
# define the method for calculating average L1 Loss and R2 Score of given model
def model_loss(model, dataset, train=False, optimizer=None):
    # first calculated for the batches and at the end get the average
    performance = L1Loss()
    # score_metric = R2Score()

    avg_loss = 0
    avg_score = 0
    count = 0

    for input, output in iter(dataset):
        # get predictions of the model for training set
        predictions = model.feed(input)

        # calculate loss of the model
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

        # store the loss and update values
        avg_loss += loss.item()
        # avg_score += score
        count += 1

    return avg_loss / count, avg_score / count


# + id="9wWr4SlYagXF" colab_type="code" outputId="181452fd-b52f-46f0-b144-f6fa9940d8d1" colab={"base_uri": "https://localhost:8080/", "height": 1000}
# define the number of epochs
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
    # model is cycled through the batches
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

# + id="u8HdqecTaj4V" colab_type="code" outputId="6bd7045e-115c-4d2e-d98c-b0559e9aa0de" colab={"base_uri": "https://localhost:8080/", "height": 50}
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
