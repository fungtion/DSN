import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
from data_loader import GetLoader
from torchvision import datasets
from model_compat import DSN
import torchvision.utils as vutils


def test(epoch, name):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 64
    image_size = 28

    ###################
    # load data       #
    ###################

    img_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    model_root = 'model'
    if name == 'mnist':
        mode = 'source'
        image_root = os.path.join('dataset', 'mnist')
        dataset = datasets.MNIST(
            root=image_root,
            train=False,
            transform=img_transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

    elif name == 'mnist_m':
        mode = 'target'
        image_root = os.path.join('dataset', 'mnist_m', 'mnist_m_test')
        test_list = os.path.join('dataset', 'mnist_m', 'mnist_m_test_labels.txt')

        dataset = GetLoader(
            data_root=image_root,
            data_list=test_list,
            transform=img_transform
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )

    else:
        print 'error dataset name'

    ####################
    # load model       #
    ####################

    my_net = DSN()
    checkpoint = torch.load(os.path.join(model_root, 'dsn_mnist_mnistm_epoch_' + str(epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    ####################
    # transform image  #
    ####################


    def tr_image(img):

        img_new = (img + 1) / 2

        return img_new


    len_dataloader = len(dataloader)
    data_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        data_input = data_iter.next()
        img, label = data_input

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(input_img).copy_(img)
        class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        result = my_net(input_data=inputv_img, mode='source', rec_scheme='share')
        pred = result[3].data.max(1, keepdim=True)[1]

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='all')
        rec_img_all = tr_image(result[-1].data)

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='share')
        rec_img_share = tr_image(result[-1].data)

        result = my_net(input_data=inputv_img, mode=mode, rec_scheme='private')
        rec_img_private = tr_image(result[-1].data)

        if i == len_dataloader - 2:
            vutils.save_image(rec_img_all, name + '_rec_image_all.png', nrow=8)
            vutils.save_image(rec_img_share, name + '_rec_image_share.png', nrow=8)
            vutils.save_image(rec_img_private, name + '_rec_image_private.png', nrow=8)

        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct * 1.0 / n_total

    print 'epoch: %d, accuracy of the %s dataset: %f' % (epoch, name, accu)
