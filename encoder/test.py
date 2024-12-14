from Encoder import EncoderLayer
import torch

# tips：下面是关于encoder的测试
d_model = 8
num_heads = 4
d_hidden = 128
seq_len = 10
batch_size = 2

# Initialize the EncoderLayer
encoder_layer = EncoderLayer(d_model, num_heads, d_hidden)

x = torch.rand(batch_size, seq_len, d_model)

# Forward pass through the encoder layer
output = encoder_layer(x)

# Verify the output shape
print("Input shape:", x.shape,x)
print("Output shape:", output.shape,output)
