import torch
from source.models.mobilenetv2 import MobileNetV2
import numpy as np
import cv2
def load_model(path='result/runs_m/last.pt',device='cuda:0'):
    model = MobileNetV2([76])
    model.load_state_dict(torch.load(path)['state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict_merge_model(model, img):
    height, width, _ = img.shape
    delta = height - width 
    if delta > 0:
        img_ = np.pad(img,[[0,0],[delta//2,delta//2],[0,0]], mode='constant',constant_values =255)
    else:
        img_ = np.pad(img,[[-delta//2,-delta//2],[0,0],[0,0]], mode='constant',constant_values =255)
    img_ = cv2.resize(img_, (224, 224))
    img_ = np.transpose(img_,[2,0,1])
    img_ = np.expand_dims(img_, axis=0)
    img_ = img_.astype('float') / 255.
    img_ = torch.Tensor(img_).to(device)
    output = model.predict(img_).detach().cpu().numpy()
    # return output
    return np.argmax(output,axis=-1), np.max(output,axis=-1)


if __name__ =='__main__':
    device = torch.device('cuda:0')
    model = load_model(path='result/runs_m/last.pt',device=device)
    img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)
    out = predict_merge_model(model,img)
    print(out)