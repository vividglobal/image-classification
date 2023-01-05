import torch
import logging 
import platform

LOGGER = logging.getLogger('__main__.'+__name__)

def select_device(device='',model_name=''):
    s = f' {model_name.upper()} author:dovietchinh 💥💥💥💥💥  torch {torch.__version__}'
    device = str(device).lower().replace('cuda:','').strip()
    cpu = device=='cpu'
    cuda = not cpu and torch.cuda.is_available()
    devices = device.split(',') if device else '0'
    if cuda:
        for i,d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    LOGGER.info(s.encode().decode('ascii','ignore') if platform.system()=='Windows' else s)
    return torch.device('cuda:0' if cuda else 'cpu')
    
def loadingImageNetWeight(model,name,model_urls):
    import io
    import requests
    if hasattr(model,'name'):
        name = getattr(model,'name')
    assert name in model_urls, f"please model.name must be in list {list(model_urls.keys())}"
    imagenet_state_dict = torch.load(io.BytesIO(requests.get(model_urls[name]).content))
    my_state_dict = model.state_dict()
    temp = {}
    for k,v in imagenet_state_dict.items():
        if k in my_state_dict:
            if v.shape==my_state_dict.get(k).shape:
                temp[k]=v
    my_state_dict.update(temp)
    model.load_state_dict(my_state_dict)
    return model