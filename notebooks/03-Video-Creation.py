
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import skvideo.io
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm_notebook

from fast_neural_style.transformer_net import TransformerNet
from fast_neural_style.utils import recover_image, tensor_normalizer


# In[2]:


preprocess = transforms.Compose([
    transforms.ToTensor(),
    tensor_normalizer()
])


# In[3]:


transformer = TransformerNet()


# ## Low Resolution GIF Animation

# Convert gif file to video file: 
# ```
# ffmpeg -f gif -i cat.gif cat.mp4
# ```

# In[ ]:


skvideo.io.ffprobe("videos/cat.mp4")


# In[2]:


transformer.load_state_dict(torch.load("../models/udine_10000.pth"))


# In[ ]:


frames = []
frames_orig = []
videogen = skvideo.io.vreader("videos/cat.mp4")
for frame in videogen:
    frames_orig.append(Image.fromarray(frame))
    frames.append(recover_image(transformer(
        Variable(preprocess(frame).unsqueeze(0), volatile=True)).data.numpy())[0])


# In[ ]:


Image.fromarray(frames[3])


# In[ ]:


writer = skvideo.io.FFmpegWriter("cat.mp4")# tuple([len(frames)] + list(frames[0].shape)))
for frame in frames:
    writer.writeFrame(frame)
writer.close()


# ## Higher Resolution Videos

# In[4]:


skvideo.io.ffprobe("../videos/obama.mp4")


# Switch to GPU:

# In[4]:


transformer.cuda()
BATCH_SIZE = 2


# In[5]:


transformer.load_state_dict(torch.load("../models/mosaic_10000.pth"))


# In[6]:


batch = []
videogen = skvideo.io.FFmpegReader("../videos/cod-2.mp4", {"-ss": "00:05:00", "-t": "00:01:00"})
writer = skvideo.io.FFmpegWriter("../videos/cod-clip-noise.mp4")
try:
    with torch.no_grad():
        for frame in tqdm_notebook(videogen.nextFrame()):
            batch.append(preprocess(frame).unsqueeze(0))
            if len(batch) == BATCH_SIZE:
                for frame_out in recover_image(transformer(
                    torch.cat(batch, 0).cuda()).cpu().numpy()):
                    writer.writeFrame(frame_out)
                batch = []
except RuntimeError as e:
    print(e)
    pass
writer.close()


# In[8]:


transformer.load_state_dict(torch.load("../models/udine_10000_unstable.pth"))


# In[9]:


batch = []
videogen = skvideo.io.FFmpegReader("../videos/cod-2.mp4", {"-ss": "00:05:00", "-t": "00:01:00"})
writer = skvideo.io.FFmpegWriter("../videos/cod-clip.mp4")
try:
    with torch.no_grad():
        for frame in tqdm_notebook(videogen.nextFrame()):
            batch.append(preprocess(frame).unsqueeze(0))
            if len(batch) == BATCH_SIZE:
                for frame_out in recover_image(transformer(
                    torch.cat(batch, 0).cuda()).cpu().numpy()):
                    writer.writeFrame(frame_out)
                batch = []
except RuntimeError as e:
    pass
writer.close()

