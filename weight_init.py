from torch import nn

def weight_init(m: nn.Module):
    """
    Initialize the weights of the generator and discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Convolutional layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Batch normalization layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)