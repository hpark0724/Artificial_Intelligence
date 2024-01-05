import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # train_set
    train_set = datasets.FashionMNIST('./data', train= True, 
                                      download = True, transform = transform)
    # test_set
    test_set = datasets.FashionMNIST('./data', train = False, transform = transform)

    # obtain DataLoader by setting shuffle = True for train loader, setting batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle = True)
    # obtain DataLoader by setting shuffle = False for test loader, setting batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)

    # return DataLoader with training set
    if training:
        return train_loader
    # return DataLoader with test set
    else:
        return test_loader    



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        # reformat data
        nn.Flatten(),
        # 28 * 28 grayscale image
        # a dense layer with 128 nodes and a ReLU
        nn.Linear(784, 128),
        # ReLU activation
        nn.ReLU(),
        # a dense layer with 64 nodes and a ReLU
        nn.Linear(128, 64),
        # ReLU activation
        nn.ReLU(),
        # a dense layer with 10 nodes, 
        nn.Linear(64,10)
    )
    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    # use cross-entropy loss nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # set up an optimizer using SGD with a learning rate of 0.001 and momemtum of 0.9
    optimizer = optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)

    # the for loop iterates over epochs
    for epoch in range(T):
        total_num = 0
        correct_num = 0
        loss_sum = 0
        # the for loop itertes over batches of (images, labels) pairs
        # from the train DataLoader
        for data, label in train_loader:
            # reset the gradient to zero
            optimizer.zero_grad()
            # put the data in the model and output the
            # predicted label result
            prediction = model(data)
            # calculate the loss of the predicted label
            # and the actual label
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            # the class with highest score as our prediction
            _, prediction = torch.max(prediction.data, 1)

            # the total number of data (images)
            total_num += data.size(0)
            # the correct number of labels
            correct_num += (prediction == label).sum().item()
            # the loss sum
            loss_sum += loss.item() * data.size(0)
        
        # calculate the percentage of accuray 
        accuracy = correct_num / total_num * 100
        # calculate the loss of the predicted label
        loss = loss_sum / total_num
        #print(f"Train Epoch: {epoch}\tAccuracy: {correct_num}/{total_num} ({accuracy:.2f}%)\tLoss: {loss:.3f}")
        print(f"Train Epoch: {epoch}\tAccuracy: {correct_num}/{total_num} ({accuracy:.2f}%) Loss: {loss:.3f}")



def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    with torch.no_grad():
        total_num = 0
        correct_num = 0
        loss_sum = 0
        for data, label in test_loader:
            # put the data in the model and output the
            # predicted label result
            prediction = model(data)
            # calculate the loss of the predicted label
            # and the actual label            
            loss = criterion(prediction, label) 
            
            _, label_pred = torch.max(prediction.data, 1)

            # the total number of data (images)
            total_num += data.size(0)
            # the correct number of labels
            correct_num += (label_pred == label).sum().item()
            # the loss sum
            loss_sum += loss.item() * data.size(0)

        # calculate the percentage of accuray 
        accuracy = correct_num / total_num * 100
        # calculate the loss of the predicted label
        loss = loss_sum / total_num

        if show_loss:
            print(f"Average loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}%")
        else:
            print(f"Accuracy: {accuracy:.2f}%")



def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    # a test image set at the given index
    single_image = test_images[index]
    # output logits
    logits = model(single_image)
    # convert the output of the final dense layer to probabilities 
    probability = F.softmax(logits, dim = -1)
    # convert the probability to a one-demensional array and 
    # store the pair of index and probability into the list
    pair = [(index, prob) for index, prob in enumerate(probability.flatten())]
    # sort the pair in the list in descending order
    index_sorted = sorted(pair, key = lambda x: x[1], reverse = True)
    # loop top three predicted classes and its probabilities 
    # and print them 
    for i in range(3):
        class_index, class_prob = index_sorted[i]
        print(f"{class_names[class_index]}: {class_prob * 100:.2f}%")




    

    



    





# if __name__ == '__main__':
#     '''
#     Feel free to write your own test code here to exaime the correctness of your functions. 
#     Note that this part will not be graded.
#     '''
#     criterion = nn.CrossEntropyLoss()
