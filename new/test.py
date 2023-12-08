import torch
import pandas as pd
from dataset import DataSet
from torch.utils.data import DataLoader
from model import SBConvNext, SBResNet
from config import get_config
import os
from PIL import ImageFile
import warnings
from tqdm import tqdm  # 导入 tqdm

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_model(model_path, device, model_type):
    if model_type == 'convnext':
        model = SBConvNext()
    elif model_type == 'resnet':
        model = SBResNet()
    else:
        raise ValueError("Invalid model type. Choose either 'convnext' or 'resnet'.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def test_model(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    probabilities = []

    with torch.no_grad():
        # 使用 tqdm 包裹测试加载器，以显示进度条
        for images, _ in tqdm(test_loader, desc='Testing', leave=True):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return predictions, probabilities

if __name__ == '__main__':
    args = get_config()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')

    # Load model
    model_path = os.path.join(args.model_path, 'fold_3_best_convnext_model.pth')  # Replace with your model file name
    model = load_model(model_path, device, args.model)

    # Load test data
    test_image_dir = '../DRAC2022_TaskB/data/1. Original Images/b. Testing Set/'
    test_image_path_list = os.listdir(test_image_dir)
    test_image_path_list.sort(key=lambda x:int(x.split('.')[0]))

    test_image_paths = []
    for name in test_image_path_list:
        test_image_paths.append(test_image_dir + name)

    test_labels = [-1] * len(test_image_paths)

    test_dataset = DataSet(test_image_paths, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Perform test
    predictions, probabilities = test_model(test_loader, model, device)

    # Extract only file names for saving in the CSV
    file_names = [os.path.basename(path) for path in test_image_paths]

    # Save results to CSV file
    results = pd.DataFrame({
        'case': file_names,
        'class': predictions,
        'P0': [prob[0] for prob in probabilities],
        'P1': [prob[1] for prob in probabilities],
        'P2': [prob[2] for prob in probabilities]
    })

    results.to_csv('test_results.csv', index=False)
    print("Test results saved to 'test_results.csv'")