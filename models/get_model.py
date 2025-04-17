from models.basenet import ResBase, VGGBase

def get_base_model(net, num_class, temp=0.05, top=False, norm=True, use_bottleneck=False, bottleneck_dim=256):
    if "resnet" in net:
        model_g = ResBase(net, top=top, use_btnk=use_bottleneck, btnk_dim=bottleneck_dim)
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
    dim = model_g.dim
    if top:
        dim = 1000
    if use_bottleneck:
        dim = bottleneck_dim
    print("selected network %s"%net)
    return model_g, dim