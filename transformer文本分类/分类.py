import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerModel
import os
import random
from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd

# 数据预处理
def prepare_data(max_seq_length):
    # 加载本地的 Parquet 文件
    train_data = pd.read_parquet("data/train-00000-of-00001.parquet")
    test_data = pd.read_parquet("data/test-00000-of-00001.parquet")
    
    # 转为 Hugging Face Dataset
    train_data = Dataset.from_pandas(train_data)
    test_data = Dataset.from_pandas(test_data)
    
    train_data = train_data.shuffle(seed=42)
    test_data = test_data.shuffle(seed=42)
    
    # 加载分词器
    local_tokenizer_dir = "./tokenizer_files/"
    tokenizer = AutoTokenizer.from_pretrained(
        local_tokenizer_dir,
        trust_remote_code=True,  # 如果使用的分词器有自定义代码，需要启用此选项
        truncation_side='right',  # 设置分词器的截断侧
        padding_side='right'  # 设置分词器的填充侧
    )
    
    # 定义编码函数
    def encode_data(data):
        encodings = tokenizer(
            data['text'], padding=True, truncation=True, max_length=max_seq_length, return_tensors="pt"
        )
        labels = torch.tensor([1 if label == 1 else 0 for label in data['label']])
        return encodings, labels

    # 编码数据
    train_encodings, train_labels = encode_data(train_data)
    test_encodings, test_labels = encode_data(test_data)
    
    return train_encodings, train_labels, test_encodings, test_labels

# 数据加载
vocab_size = 30522
nclass = 2
d_model = 128
num_heads = 4
num_decoder_layers = 2
dim_feedforward = 256
max_seq_length = 128
dropout = 0.1
batch_size = 32
learning_rate = 5e-4
num_epochs = 10
model_save_path = "./transformer_model/"
print("#####data process start######")
train_encodings, train_labels, test_encodings, test_labels = prepare_data(max_seq_length)
print("#####data process over######")

train_zeros = (train_labels == 0).sum().item()
train_ones = (train_labels == 1).sum().item()
train_total = len(train_labels)
train_zero_ratio = train_zeros / train_total
train_one_ratio = train_ones / train_total
print(f"Train Labels Distribution: 0 -> {train_zeros} ({train_zero_ratio:.2%}), 1 -> {train_ones} ({train_one_ratio:.2%})")

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    max_seq_length=max_seq_length,
    nclass=nclass,
    dropout=dropout
).to(device)

# 损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建保存目录
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

print("#####training strart######")
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(train_labels), batch_size):
        input_ids = train_encodings['input_ids'][i:i+batch_size].to(device)
        attention_mask = train_encodings['attention_mask'][i:i+batch_size].to(device)
        labels = train_labels[i:i+batch_size].to(device)
        outputs, _ = model(input_ids, input_ids, attention_mask.unsqueeze(1).unsqueeze(2), None)
        logits = outputs[:, 0, :]
        random_number = random.random() 
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 保存模型状态字典
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, model_save_path+f"{epoch}.pt")
    print(f"Model saved after epoch {epoch+1} to {model_save_path}")

checkpoint = torch.load("/home/rzzhang/trans/transformer_model/9.pt", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

model.eval()
correct = 0
total = 0
print("#####evaluation start######")
with torch.no_grad():
    for i in range(0, len(test_labels), batch_size):
        input_ids = test_encodings['input_ids'][i:i+batch_size].to(device)
        attention_mask = test_encodings['attention_mask'][i:i+batch_size].to(device)
        labels = test_labels[i:i+batch_size].to(device)
        # 模型前向传播
        outputs, _ = model(input_ids, input_ids, attention_mask.unsqueeze(1).unsqueeze(2), None)
        # 获取 logits（每个样本的 [负类分数, 正类分数]）
        logits = outputs[:, 0, :]
        predicted = torch.argmax(logits, dim=1)
        # 统计正确的预测数
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# 计算准确率
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")