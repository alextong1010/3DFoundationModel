import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
import torchvision
import open_clip

def load_vision_encoder(vision_encoder_type, backbone, weights):
    if vision_encoder_type == "clip":
        print("Use OpenCLIP as vision encoder")

        clip_feature_dim_dict = {'ViT-B-32': 512,
                                'ViT-B-16': 512,
                                'ViT-L-14': 768,
                                'ViT-H-14': 1024,
                                'ViT-g-14': 1024}
        backbone_arch = backbone

        # Get a dictionary of all available models and their pretrained weights
        open_clip_model_dict = open_clip.list_pretrained()
        open_clip_model_dict = [x for x in open_clip_model_dict if x[0]==backbone_arch]

        flag = False
        for model_pair in open_clip_model_dict:
            if model_pair[1]==weights:
                flag = True
                break
        if not flag:
            raise Exception("{} is an unsupported pretrained CLIP weights!".format(weights))
        
        vision_encoder, _, transform = open_clip.create_model_and_transforms(backbone_arch, pretrained=weights) # This line already loads fine-tuned CLIP weights from local path
        
        vision_feature_dim = clip_feature_dim_dict[backbone_arch]

    elif vision_encoder_type == "dinov2":
        print("Use DinoV2 as vision encoder")

        dinov2_feature_dim_dict = {'dinov2_vits14': 384,
                                    'dinov2_vitb14': 768,
                                    'dinov2_vitl14': 1024,
                                    'dinov2_vitg14': 1536}

        backbone_arch = backbone.lower().replace('-', '')

        # List available models and weights from the dinov2 repository
        dinov2_model_list = torch.hub.list('facebookresearch/dinov2')

        flag = False
        for model_weight in dinov2_model_list:
            if model_weight==backbone_arch:
                flag = True
                break
        if not flag:
            raise Exception("{} is an unsupported pretrained dinov2 weights!".format(backbone))

        vision_encoder = torch.hub.load('facebookresearch/dinov2', backbone_arch) #load the backbone
        
        vision_feature_dim = dinov2_feature_dim_dict[backbone_arch]

        # check: https://github.com/facebookresearch/dinov2/tree/main?tab=readme-ov-file#pretrained-heads---image-classification
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise Exception("vision_encoder is not recognized!")
        
    return vision_encoder, transform, vision_feature_dim