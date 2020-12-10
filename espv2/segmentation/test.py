from cnn import SegmentationModel as net
from torchsummary import summary

model = net.EESPNet_Seg(args.classes, s=args.s, pretrained=args.pretrained, gpus=num_gpus)
summary(model, (3, 120, 320))