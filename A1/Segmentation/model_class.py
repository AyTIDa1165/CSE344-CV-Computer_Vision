import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class SegNet_Encoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()

        #SegNet Architecture
        #Takes input of size in_chn = 3 (RGB images have 3 channels)
        #Outputs size label_chn (N # of classes)

        #ENCODING consists of 5 stages
        #Stage 1, 2 has 2 layers of Convolution + Batch Normalization + Max Pool respectively
        #Stage 3, 4, 5 has 3 layers of Convolution + Batch Normalization + Max Pool respectively

        #General Max Pool 2D for ENCODING layers
        #Pooling indices are stored for Upsampling in DECODING layers

        self.in_chn = in_chn
        self.out_chn = out_chn

        self.MaxEn = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
    def forward(self,x):
        #ENCODE LAYERS
        #Stage 1
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        #Stage 2
        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        #Stage 3
        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        #Stage 4
        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        #Stage 5
        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        return x,[ind1,ind2,ind3,ind4,ind5],[size1,size2,size3,size4,size5]
    

class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn

        #implement the architecture.
        self.MaxDec = nn.MaxUnpool2d(kernel_size=2,stride=2)

         # stage 5:
        # Max Unpooling: Upsample using ind5 to size4
        # Channels: 512 → 512 → 512 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm
        self.ConvDec51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDec51 = nn.BatchNorm2d(512, momentum=BN_momentum) 
        self.ConvDec52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDec52 = nn.BatchNorm2d(512, momentum=BN_momentum) 
        self.ConvDec53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDec53 = nn.BatchNorm2d(512, momentum=BN_momentum) 
        
        #stage 4:
        # Max Unpooling: Upsample using ind4 to size3
        # Channels: 512 → 512 → 256 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm
        self.ConvDec41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDec41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDec42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDec42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDec43 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDec43 = nn.BatchNorm2d(256, momentum=BN_momentum)

        # Stage 3:
        # Max Unpooling: Upsample using ind3 to size2
        # Channels: 256 → 256 → 128 (3 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm
        self.ConvDec31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDec31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDec32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDec32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDec33 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDec33 = nn.BatchNorm2d(128, momentum=BN_momentum)

        # Stage 2:
        # Max Unpooling: Upsample using ind2 to size1
        # Channels:  128 → 64 (2 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after each batch norm
        self.ConvDec21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDec21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDec22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDec22 = nn.BatchNorm2d(64, momentum=BN_momentum)

        # Stage 1:
        # Max Unpooling: Upsample using ind1
        # Channels: 64 → out_chn (2 convolutions)
        # Batch Norm: Applied after each convolution
        # Activation: ReLU after the first convolution, no activation after the last one 
        self.ConvDec11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDec11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDec12 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)
        self.BNDec12 = nn.BatchNorm2d(self.out_chn, momentum=BN_momentum)

        #For convolution use kernel size = 3, padding =1 
        #for max unpooling use kernel size=2 ,stride=2 

    def forward(self,x,indexes,sizes):
        ind1,ind2,ind3,ind4,ind5=indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]
        size1,size2,size3,size4,size5=sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]
        
        #stage5
        x = self.MaxDec(x, ind5, size4)
        x = F.relu(self.BNDec51(self.ConvDec51(x)))
        x = F.relu(self.BNDec52(self.ConvDec52(x)))
        x = F.relu(self.BNDec53(self.ConvDec53(x)))

        #stage4
        x = self.MaxDec(x, ind4, size3)
        x = F.relu(self.BNDec41(self.ConvDec41(x)))
        x = F.relu(self.BNDec42(self.ConvDec42(x)))
        x = F.relu(self.BNDec43(self.ConvDec43(x)))

        #stage3
        x = self.MaxDec(x, ind3, size2)
        x = F.relu(self.BNDec31(self.ConvDec31(x)))
        x = F.relu(self.BNDec32(self.ConvDec32(x)))
        x = F.relu(self.BNDec33(self.ConvDec33(x)))

        #stage2
        x = self.MaxDec(x, ind2, size1)
        x = F.relu(self.BNDec21(self.ConvDec21(x)))
        x = F.relu(self.BNDec22(self.ConvDec22(x)))

        #stage1
        x = self.MaxDec(x, ind1, torch.Size([size1[0], 32, 360, 480]))
        x = F.relu(self.BNDec11(self.ConvDec11(x)))
        x = self.ConvDec12(x)
        return x    

class SegNet_Pretrained(nn.Module):
    def __init__(self,encoder_weight_pth,in_chn=3, out_chn=32):
        super(SegNet_Pretrained, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder=SegNet_Encoder(in_chn=self.in_chn,out_chn=self.out_chn)
        self.decoder=SegNet_Decoder(in_chn=self.in_chn,out_chn=self.out_chn)
        encoder_state_dict = torch.load(encoder_weight_pth,weights_only=True)

        # Load weights into the encoder
        self.encoder.load_state_dict(encoder_state_dict)

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        x,indexes,sizes=self.encoder(x)
        x=self.decoder(x,indexes,sizes)
        return x

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=32):
        super(DeepLabV3, self).__init__()
        # TODO: Initialize DeepLabV3 model here using pretrained=True
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        #  should be a Conv2D layer with input channels as 256 and output channel as num_classes using a stride of 1, and kernel size of 1.
        self.model.classifier[4] = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1)
       
    def forward(self, x):
        return self.model(x)['out']