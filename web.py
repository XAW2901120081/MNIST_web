import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 加载模型
model = Net()
model.load_state_dict(torch.load("web/model.pth"))
model.eval()

# 定义图像预处理函数
def image_loader(image):
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)
    return image

# 定义预测函数
def predict_image(image):
    image = image_loader(image)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# 创建 Streamlit 应用
st.title('手写数字识别')

uploaded_file = st.file_uploader("选择一个手写数字图像", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='上传的图像', use_column_width=True)
    st.write("正在预测...")
    
    prediction = predict_image(image)
    st.write(f"预测的数字是: {prediction}")
