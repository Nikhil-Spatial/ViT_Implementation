## **Vision Transformer Paper Implementation**

In this Vision Transformer implementation, I developed all of the key components of the model from scratch, such as the Multi-head Self-attention mechanism and Transformer layers. By scratch, I mean only using PyTorch
components and not the ready-to-use modules such as *Torch.nn.TransformerEncoderLayer*. 

I used the original Vision Transformer paper's model architecture:
<img src="assets/loss_function.png" width="500">

The model was trained on the CIFAR-10 dataset with these specific configurations:
- Image patch width/height: 4 pixels
- Patch embeddings: 512 dimensions
- Heads in Multi-head Self-attention layers: 4
- Number of Transformer layers in the Transformer Encoder: 8
- Dropout probability: 0.2

Its accuracy on the test dataset is --.--%. 
