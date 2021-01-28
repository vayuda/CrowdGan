from PIL import Image
import torch
import numpy as np
from itertools import product
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
import IPython
import os, pickle
from models import *
from Auxilary_Functions import *


def train_model(model, optimizer, data, num_epochs, lossf, test_x, test_y, is_crowd=False, print_progress=2,
                device="cuda:0", run_name='training:'):
    accs = []
    if print_progress == 1:
        for ep in trange(num_epochs, desc=run_name):
            batch_n = 0
            for x, y in data:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                out = model(x).to(device)
                loss = masked_cross_entropy(out, y, lossf) if is_crowd else lossf(out, y)
                loss.backward()
                optimizer.step()
                batch_n = batch_n + 1
            acc = eval_model(model, test_x, test_y, crowd_eval=is_crowd)
            accs.append(acc)
    else:
        for ep in range(num_epochs):
            batch_n = 0
            for x, y in data:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                out = model(x).to(device)
                loss = masked_cross_entropy(out, y, lossf) if is_crowd else lossf(out, y)
                loss.backward()
                optimizer.step()
                batch_n = batch_n + 1
            acc = eval_model(model, test_x, test_y, crowd_eval=is_crowd)
            accs.append(acc)
            if print_progress == 2:
                print(f"epoch {ep + 1}/{num_epochs}\tTest accuracy: {acc}")
    return accs


def eval_model(model, test_data, test_labels, crowd_eval=False, device="cuda:0"):
    preds = model(test_data.to(device), True) if crowd_eval else model(test_data.to(device))
    preds = torch.argmax(preds, dim=1).detach()
    preds = np.array(preds.cpu()) if device == "cuda:0" else np.array(preds)
    accuracy = 1.0 * np.sum(preds == test_labels) / len(test_labels)
    return accuracy


def create_one_hots(n_annotators, n_classes):
    crowd_labels = np.load('labelme/prepared/answers.npy')
    crowd_labels_onehot = torch.zeros(crowd_labels.shape[0], n_annotators, n_classes)
    for i, instancelabels in enumerate(crowd_labels):
        for j, label in enumerate(instancelabels):
            new_label = torch.zeros(n_classes)
            if label >= 0:
                new_label[label] = 1
            else:
                new_label -= 1
            crowd_labels_onehot[i][j] = new_label
    torch.save(crowd_labels_onehot, "crowd_labels_onehot.pt")


def crowd_layer_demo():
    n_classes = 8
    n_annotators = 59

    num_workers = 2
    ngpu = 1
    deterministic = True

    num_epochs = 10
    batch_size = 64
    lr = 0.0002
    betas = (0.5, 0.999)

    manualSeed = 999 if deterministic else random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    train_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_train_vgg16.npy'))
    ground_truth_labels = torch.from_numpy(np.load('labelme/prepared/labels_train.npy')).type(torch.int64)
    crowd_labels = torch.from_numpy(np.load('labelme/prepared/answers.npy'))

    test_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_test_vgg16.npy'))
    test_labels = np.load('labelme/prepared/labels_test.npy')

    print(f'train data size: {train_data_vgg16.size()} ',
          f'\nlabels size: {ground_truth_labels.size()}'
          f'\ncrowd sourced labels size: {crowd_labels.size()}'
          f'\ntest data size: {test_data_vgg16.size()}')

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    base = BaseModel((4, 4, 512), n_classes).to(device)
    crowd = CrowdNetwork((4, 4, 512), n_classes, 59).to(device)

    base.apply(weights_init)
    crowd.apply(weights_init)

    optimizer_base = Adam(base.parameters(), lr=lr, betas=betas)
    optimizer_crowd = Adam(crowd.parameters(), lr=lr, betas=betas)

    ce_loss = nn.CrossEntropyLoss(reduction='mean')

    train_data_base = list(zip(train_data_vgg16, ground_truth_labels))
    train_data_crowd = list(zip(train_data_vgg16, crowd_labels))

    dataloader_base = DataLoader(train_data_base, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, drop_last=True)
    dataloader_crowd = DataLoader(train_data_crowd, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)

    losses_base = train_model(base, optimizer_base, dataloader_base, num_epochs,
                              ce_loss, test_data_vgg16, test_labels, is_crowd=False)
    losses_crowd = train_model(crowd, optimizer_crowd, dataloader_crowd, num_epochs,
                               ce_loss, test_data_vgg16, test_labels, is_crowd=True)

    plt.figure()
    plt.title('Test set Accuracy')
    plt.plot(losses_base, label='base')
    plt.plot(losses_crowd, label='crowd')
    plt.legend()
    plt.xlabel('Update Iterations')
    plt.ylabel('Accuracy')
    plt.show()


def hyper_parameter_grid_search(num_trials=1, existing_model=True):
    n_classes = 8
    n_annotators = 59

    num_workers = 2
    ngpu = 1
    seed = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    num_epochs = [15]
    batch_size = [256]
    lr = [1e-4]
    beta1 = [0.7]

    configurations = list(product(num_epochs, batch_size, lr, beta1))
    runs = len(configurations)
    print(f"testing {runs} configurations {num_trials} times")

    train_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_train_vgg16.npy'))
    crowd_labels = torch.from_numpy(np.load('labelme/prepared/answers.npy'))

    test_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_test_vgg16.npy'))
    test_labels = np.load('labelme/prepared/labels_test.npy')

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    train_data_crowd = list(zip(train_data_vgg16, crowd_labels))

    best_model = None
    best_acc = 0
    if existing_model:
        best_model = torch.load("temp.pt")
        best_acc = eval_model(best_model, test_data_vgg16, test_labels, crowd_eval=True)
        print(f'starting model accuracy: {best_acc}')

    for configuration in configurations:
        num_epochs, batch_size, lr, beta1 = configuration
        for trial in range(num_trials):
            random.seed(trial * 100 + 100)
            torch.manual_seed(trial * 100)
            crowd = CrowdNetwork((4, 4, 512), n_classes, n_annotators).to(device)
            crowd.apply(weights_init)
            optimizer_crowd = Adam(crowd.parameters(), lr=lr, betas=(beta1, 0.999))

            dataloader_crowd = DataLoader(train_data_crowd, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, drop_last=True)
            train_model(crowd, optimizer_crowd, dataloader_crowd, num_epochs,
                        ce_loss, test_data_vgg16, test_labels,
                        is_crowd=True, print_progress=1, run_name=str(configuration))
            acc = eval_model(crowd, test_data_vgg16, test_labels, crowd_eval=True)
            print(acc)
            if acc > best_acc:
                best_acc = acc
                best_model = crowd
                print(f'\nSeed: {trial * 100} num_epochs: {num_epochs} batch size: {batch_size} '
                      f'lr: {lr} beta1:{beta1} new best accuracy: {acc:.4f}')
                torch.save(best_model, "temp.pt")


def CFGAN_train():
    n_classes = 8
    n_annotators = 59

    num_workers = 2
    ngpu = 1
    deterministic = True
    seed = 100 if deterministic else random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    prefix = "v2.4"
    if not os.path.exists(f"results/{prefix}"):
        os.makedirs(f"results/{prefix}")
    trials = 5
    num_epochs = [100]
    batch_size = [1024]
    lr_d = [1e-3]
    lr_g = [1e-4]
    num_hidden = [32]
    n_d_train = [5]
    models = [DiscriminatorLatest]
    configurations = list(product(num_epochs, num_hidden, models, batch_size, lr_g, lr_d, n_d_train))
    train_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_train_vgg16.npy'))
    crowd_labels = torch.load("crowd_labels_onehot.pt")
    cl = torch.from_numpy(np.load('labelme/prepared/answers.npy')).type(torch.float32)
    discriminator = torch
    test_data_vgg16 = torch.from_numpy(np.load('labelme/prepared/data_test_vgg16.npy'))
    test_labels = np.load('labelme/prepared/labels_test.npy')

    g_weights = 'models/crowd_generator.pt'  # pre-trained generator

    generator = torch.load(g_weights).to(device)
    # generator = CrowdNetwork((4, 4, 512), n_classes, n_annotators).to(device)
    # generator.apply(weights_init)

    best_acc = eval_model(generator, test_data_vgg16, test_labels, crowd_eval=True)
    print("\nstarting test accuracy: ", best_acc)

    train_data_crowd = list(zip(train_data_vgg16, crowd_labels))

    bce_loss = nn.BCELoss()
    best_config = None
    print(f'testing {len(configurations)} configurations {trials} times')
    for configuration in configurations:
        for trial in range(trials):
            counter = 0
            seed = 100 + 1000*trial
            random.seed(seed)
            torch.manual_seed(seed)

            num_epochs, num_hidden, model, batch_size, lr_g, lr_d, n_d_train = configuration

            discriminator = model(batch_size, num_hidden, n_annotators, n_classes).to(device)
            discriminator.apply(weights_init)
            # discriminator = torch.load("models/discriminator.pt")

            # numbers are in order of: learning rate discriminator, generator, seed, batchsize
            title = f'model_{str(discriminator)}_{lr_d:.3f}_{lr_g:.3f}_{seed}_{batch_size}'

            dataloader = DataLoader(train_data_crowd, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers, drop_last=True)

            optimizer_g = Adam(generator.parameters(), lr=lr_g)
            optimizer_d = Adam(discriminator.parameters(), lr=lr_d)

            real_target = torch.ones(n_annotators,1).to(device)
            fake_target = torch.zeros(n_annotators,1).to(device)

            loss_d, loss_g, test_acc, acc_authentic, acc_generated = [], [], [], [], []
            for ep in trange(num_epochs, desc=title):
                batch_n = 0
                for x, y in dataloader:
                    optimizer_d.zero_grad()
                    x = x.to(device)
                    y = y.to(device)
                    g_out = generator(x).to(device)
                    g_out = g_out.permute(2, 1, 0)

                    y = y.permute(1, 2, 0)
                    mask = y != -1
                    g_out = torch.softmax(g_out, 1)
                    masked_g_out = torch.where(mask, g_out, torch.zeros_like(g_out))
                    masked_y = torch.where(mask, y, torch.zeros_like(y))

                    adr = torch.zeros(n_d_train, n_annotators)
                    adf = torch.zeros(n_d_train, n_annotators)
                    ld = torch.zeros(n_d_train)

                    for __ in range(n_d_train):
                        d_real = discriminator(masked_y)
                        d_fake = discriminator(masked_g_out.detach())
                        e_dr = bce_loss(d_real, real_target)
                        e_df = bce_loss(d_fake, fake_target)
                        e_d = e_dr + e_df
                        e_d.backward()
                        adr[__] = d_real.view(1,-1)
                        adf[__] = d_fake.view(1,-1)
                        ld[__] = e_d.item()

                    loss_d.append((counter, torch.mean(ld)))
                    acc_authentic.append(torch.mean(torch.where(adr > .8, torch.ones_like(adr), torch.zeros_like(adr))))
                    acc_generated.append(torch.mean(torch.where(adf < .2, torch.ones_like(adf), torch.zeros_like(adf))))

                    optimizer_d.step()
                    if acc_authentic[-1] > .9:
                        optimizer_g.zero_grad()
                        d_fake = discriminator(masked_g_out)
                        e_g = torch.mean(torch.log(1. - d_fake))
                        # e_g = bce_loss(d_fake, real_target)
                        loss_g.append((counter, e_g.item()))
                        e_g.backward()
                        optimizer_g.step()
                    batch_n += 1
                    counter += 1
                acc = eval_model(generator, test_data_vgg16, test_labels, crowd_eval=True)
                test_acc.append(acc)

            if test_acc[-1] > best_acc:
                best_acc = test_acc[-1]
                best_config = configuration
                print('new best', configuration, seed, test_acc[-1])

            fig, (ax1, ax2, ax3) = plt.subplots(3)

            ax1.set_title(title)
            ax1.plot([i[0] for i in loss_g], [i[1] for i in loss_g], label='generator')
            ax1.plot([i[0] for i in loss_d], [i[1] for i in loss_d], label='discriminator')
            ax1.legend()
            ax1.set_xlabel('Update Iterations')
            ax1.set_ylabel('Training Loss')

            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Test Accuracy")
            ax2.plot(test_acc, marker='o', linestyle='-')

            ax3.plot(acc_generated,label='generated')
            ax3.plot(acc_authentic, label='authentic')
            ax3.set_ylabel("Discriminator Accuracy")
            ax3.set_xlabel("batch number")
            ax3.legend()

            plt.savefig(f"results/{prefix}/{title}_{trial}.png", format="png")
            torch.save(discriminator, "models/discriminator.pt")

            # with open("discriminator_output.p", 'wb+') as f:
            #     pickle.dump((d_reals, d_fakes), f)


if __name__ == '__main__':
    CFGAN_train()
    # with open("discriminator_output.p",'rb') as o:
    #     o = pickle.load(o)
    #     print(torch.mean(o[0]), torch.mean(o[1]))
    #     print(torch.std(o[0]), torch.std(o[1]))
