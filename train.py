import natthaphon
import models
import datagen
from torch import nn, optim
import json
import os


def lrstep(epoch):
    if epoch < 150:
        a = 0.05
    elif 150 < epoch < 225:
        a = 0.005
    else:
        a = 0.0005
    print(f'Epoch: {epoch+1} - returning learning rate {a}')
    return a


if __name__ == '__main__':
    model = natthaphon.Model(models.ResCift((3, 3, 3)))
    rprop = optim.SGD(model.model.parameters(), lr=0.01, momentum=0.9)
    model.compile(optimizer=rprop,
                  loss=nn.MSELoss(),
                  device='cuda'
                  )

    if os.path.isdir('/home/palm/Pictures'):
        train_datagen = datagen.SiftGenerator('/home/palm/Pictures/phuket')
        val_datagen = datagen.SiftGenerator('/home/palm/Pictures/phuket')
    else:
        train_datagen = datagen.SiftGenerator('/root/palm/DATA/mscoco/images/train2017')
        val_datagen = datagen.SiftGenerator('/root/palm/DATA/mscoco/images/val2017')

    trainloader = natthaphon.Loader(train_datagen, shuffle=True, num_workers=4)
    testloader = natthaphon.Loader(val_datagen, shuffle=False, num_workers=4)

    schedule = natthaphon.LambdaLR(rprop, lrstep)

    history = model.fit_generator(trainloader, 300, validation_data=testloader, schedule=schedule)
    with open('logs/ResCift333-1.json', 'w') as wr:
        json.dump(history, wr)
