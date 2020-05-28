import torch, time, copy, os, argparse
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as nd
from torch.autograd import Variable
import numpy as np

data_train_raw = MNIST("./data/mnist", download=True, train=True,
                   transform=transforms.Compose([
                       transforms.Resize((28, 28)),
                       transforms.ToTensor()]))

data_val_raw = MNIST("./data/mnist", train=False, download=True,
                 transform=transforms.Compose([
                     transforms.Resize((28, 28)),
                     transforms.ToTensor()]))

data_train = DataLoader(data_train_raw, batch_size=200, shuffle=True)
data_val = DataLoader(data_val_raw, batch_size=1000)

data = (data_train, data_val)

def KL(alpha, K):    
    beta = torch.ones([1, K], dtype=torch.float32)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)    
    KL_val = torch.sum((alpha - beta)*(torch.digamma(alpha)-torch.digamma(S_alpha)),dim=1,keepdim=True) + \
         torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha),dim=1,keepdim=True) + \
         torch.sum(torch.lgamma(beta),dim=1,keepdim=True) - torch.lgamma(torch.sum(beta,dim=1,keepdim=True))
    return KL_val

def eq3(evidence, target, epoch_num, K, annealing_step):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum(target * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
    annealing = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    kl = annealing * KL((alpha - 1) * (1 - target) + 1, K)
    return torch.mean(loglikelihood + kl)

def eq4(evidence, target, epoch_num, K, annealing_step):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    kl = annealing * KL((alpha - 1) * (1 - target) + 1, K)
    return torch.mean(loglikelihood + kl)

def eq5(evidence, target, epoch_num, K, annealing_step):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    annealing = torch.min(torch.tensor(
        1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))
    kl = annealing * KL((alpha - 1) * (1 - target) + 1, K)
    return torch.mean(loglikelihood + kl)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # reshape x s.t each image is flattened.
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def training_model(model, all_data, K, loss_function, optimizer, epochs=25):

    best_accuracy = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(epochs):
        print("Epoch " + str(epoch) +" of " + str(epochs - 1))
        for train_or_test in [0,1]:
            if train_or_test == 0:
                model.train()
                is_train = True
                data = all_data[0]
                stage = "training"
            else:
                model.eval()
                is_train = False
                data = all_data[1]
                stage = "validation"

            computed_loss = 0.0
            computed_corrects = 0.0
            for batch, labels in data:
                optimizer.zero_grad() # empty the gradient buffer
                # only calculate the gradient if we're training
                with torch.set_grad_enabled(is_train):
                    one_hot_encoding = torch.eye(K)[labels]
                    outputs = model(batch)
                    # the index where the maximum value appears within each row is the predicted digit
                    predicted_digits=torch.argmax(outputs, dim=1)
                    loss = loss_function(outputs, one_hot_encoding.float(), epoch, K, 10)
                    match = torch.reshape(torch.eq(predicted_digits, labels).float(), (-1, 1))
                    
                    avg_evidence_fail = torch.sum(
                        torch.sum(outputs, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                    avg_evidence_success = torch.sum(
                        torch.sum(outputs, 1, keepdim=True) * match) / torch.sum(match + 1e-20)

                    if stage == "training":
                        loss.backward()
                        optimizer.step()

                computed_loss += loss.item() * batch.size(0)
                computed_corrects += torch.sum(predicted_digits == labels.data)

            accuracy_of_total_epoch = computed_corrects.float() / len(data.dataset)

            # Measure accuracy on validation set.
            if stage == "validation" and accuracy_of_total_epoch > best_accuracy:
                best_accuracy = accuracy_of_total_epoch
                best_weights = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print("Training took " + str(round(total_time/ 60,2)) +" minutes")
    print("Best Accuracy: " + str(float(best_accuracy)))

    model.load_state_dict(best_weights)
    return model


def rotating_image_classification(img, model, threshold=0.25, dims=(28,28), c = ["black", "blue", "red", "brown", "purple", "cyan"], marker=["s", "^", "o"]*2):
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    K = 10

    scores = np.zeros((1, K))
    rot_imgs = np.zeros((dims[0], dims[1]*Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        rot_img = nd.rotate(img.reshape(*dims), deg, reshape=False).reshape(*dims)
        rot_img = np.clip(a=rot_img, a_min=0, a_max=1)
        rot_imgs[:,i*dims[1]:(i+1)*dims[1]] = 1 - rot_img

        trans = transforms.ToTensor()
        img_tens = trans(rot_img).unsqueeze_(0)
        img_var = Variable(img_tens)

        output = model(img_var)
        alpha = output + 1
        uncertainty = K / torch.sum(alpha, dim=1, keepdim=True)
        lu.append(uncertainty.mean())

        probabilities = alpha / torch.sum(alpha, dim=1, keepdim=True)
        probabilities = probabilities.flatten()

        scores += probabilities.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(probabilities.tolist())

    labels = np.arange(K)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    labels = labels.tolist()

    fig,(ax0, ax1) = plt.subplots(2, 1, figsize=(6,6))
    ax0.imshow(rot_imgs, cmap='gray')
    ax0.axis('off')

    for i in range(len(labels)):
        ax1.plot(ldeg, lp[:,i], marker=marker[i], c=c[i])
    
    labels += ['uncertainty']
    ax1.plot(ldeg, lu, marker='<', c='red')
    ax1.legend(labels)

    plt.xlim([0,Mdeg])  
    plt.xlabel('Rotation Degree')
    plt.ylabel('Classification Probability')  
    plt.tight_layout()
    plt.show()


def main():
    # path for saving model params
    dir_path = os.getcwd()+'/results'

    parser = argparse.ArgumentParser()
    # choose  whether to train or test (default option)
    train_or_test = parser.add_mutually_exclusive_group(required=True)
    train_or_test.add_argument("--train", action="store_true", default=False)
    train_or_test.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--epochs", default=10, type=int)
    # choose the loss function
    loss_function_group = parser.add_mutually_exclusive_group()
    loss_function_group.add_argument("--eqthree", action="store_true")
    loss_function_group.add_argument("--eqfour", action="store_true")
    loss_function_group.add_argument("--eqfive", action="store_true")
    args = parser.parse_args()

    if args.train:
        if args.eqthree:
            loss_function = eq3
        elif args.eqfour:
            loss_function = eq4
        elif args.eqfive:
            loss_function = eq5
        else:
            parser.error("you must specify the desired loss function: --eqthree, --eqfour or --eqfive.")

        epochs = args.epochs
        K = 10
        model = LeNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.007)

        model = training_model(model, data, K, loss_function, optimizer, epochs=epochs)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if args.eqthree:
            torch.save(state, dir_path + "/equation_three.pt")
        elif args.eqfour:
            torch.save(state, dir_path + "/equation_four.pt")
        elif args.eqfive:
            torch.save(state, dir_path + "/equation_five.pt")

    elif args.test:
        # load the learned parameters from disk
        if args.eqthree:
            saved_params = torch.load(dir_path+"/equation_three.pt")
        elif args.eqfour:
            saved_params = torch.load(dir_path+"/equation_four.pt")
        elif args.eqfive:
            saved_params = torch.load(dir_path+"/equation_five.pt")
        else:
            parser.error("you must specify the desired loss function: --eqthree, --eqfour or --eqfive.")

        model = LeNet()
        optimizer = torch.optim.Adam(model.parameters())

        model.load_state_dict(saved_params["model"])
        optimizer.load_state_dict(saved_params["optimizer"])
        # switch to testing mode
        model.eval()
        one, _ = data_val_raw[5] # the digit "1"
        rotating_image_classification(one, model)


if __name__ == "__main__":
    main()