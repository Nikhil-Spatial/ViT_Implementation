# patch embedding dimensions
D = 512

# image width and height
W = 32
H = 32

# image patch length/width
P = 4

# total number of image patch embeddings + classification token
N = int((W / P) * (H / P) + 1)

# number of images per batch for training/inference
B = 256

# dropout probability
dropout_P = 0.0
