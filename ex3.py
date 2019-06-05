import utils as ut
import nn_model as nn

def main():
    train_x, train_y = ut.read_data('train_x'), ut.read_data("train_y")
    train_x = ut.norm_data(train_x)
    train_y = ut.norm_data(train_y)

    # leaving 20% of the data for validation
    dev_size = int(len(train_x) * 0.2)
    dev_x = train_x[-dev_size:, :]
    dev_y = train_y[-dev_size:]
    train_x, train_y = train_x[:-dev_size, :], train_y[:-dev_size]

    model = nn.NNModel(784,50,10)
    model.train_nn(train_x,train_y,dev_x,dev_y,20,0.1)
    print("hey")


if __name__ == '__main__':
    main()