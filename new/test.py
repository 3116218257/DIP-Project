import torch
import pandas as pd
from dataset import DataSet
from torch.utils.data import DataLoader
from model import SBConvNext, SBResNet
from config import get_test_config
import os
from PIL import ImageFile
import warnings
from tqdm import tqdm  

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    args = get_test_config()

    args.device = 'cpu' if args.device < 0 else 'cuda:%i' % args.device
    device = torch.device(args.device)

    # Load model
    model_dict = {
        'resnet': SBResNet,
        'convnext': SBConvNext
    }

    test_num = max(len(args.test_image_dir), len(args.model_path), len(args.model))
    all_predictions = []
    all_probabilities = []

    for test in range(test_num):
        model = model_dict[args.model[test]]()
        model.load_state_dict(torch.load(args.model_path[test], map_location=device))
        model = model.to(device)

        image_dir = args.test_image_dir[test]

        # Load test data
        test_image_path_list = os.listdir(image_dir)
        test_image_path_list.sort(key=lambda x:int(x.split('.')[0]))

        test_image_paths = []
        for name in test_image_path_list:
            test_image_paths.append(os.path.join(image_dir, name))

        test_labels = [-1] * len(test_image_paths)

        test_dataset = DataSet(test_image_paths, test_labels, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Perform test
        predictions, probabilities = test_model(test_loader, model, device)

        all_predictions.append(predictions)
        all_probabilities.append(probabilities)

    # Extract only file names for saving in the CSV
    file_names = [os.path.basename(path) for path in test_image_paths]

    final_probabilities = []
    final_predictions = []

    # average probability
    for image_num in range(len(all_probabilities[0])):
        image_probabilities = []
        for label in range(3):
            label_probability = 0
            for test in range(test_num):
                label_probability += all_probabilities[test][image_num][label]
            label_probability /= test_num
            image_probabilities.append(label_probability)
        final_probabilities.append(image_probabilities)
        # final_predictions.append(image_probabilities.index(max(image_probabilities)))

    # vote
    for image_num in range(len(all_probabilities[0])):
        image_labels = [0, 0, 0]
        for test in range(test_num):
            image_labels[all_predictions[test][image_num]] += 1
        final_predictions.append(image_labels.index(max(image_labels)))

    # Save results to CSV file
    results = pd.DataFrame({
        'case': file_names,
        'class': final_predictions,
        'P0': [prob[0] for prob in final_probabilities],
        'P1': [prob[1] for prob in final_probabilities],
        'P2': [prob[2] for prob in final_probabilities]
    })

    results.to_csv('test_results.csv', index=False)
    print("Test results saved to 'test_results.csv'")