import torch
import pandas as pd
from dataset import DataSet, ValDataSet
from torch.utils.data import DataLoader
from model import SBConvNext, SBResNet
from config import get_test_config
from PIL import ImageFile
import warnings
from tqdm import tqdm  

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def evaluate_both_stages(best_model_stage0_path, best_model_stage1_path, val_dataset, model_stage0, model_stage1, device):
    model_stage0.load_state_dict(torch.load(best_model_stage0_path, map_location=device))
    model_stage0 = model_stage0.to(device)
    model_stage1.load_state_dict(torch.load(best_model_stage1_path, map_location=device))
    model_stage1 = model_stage1.to(device)

    val_dataset_stage0 = ValDataSet("data/test", val_dataset)
    val_loader_stage0 = DataLoader(val_dataset_stage0, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    model_stage0.eval()
    model_stage1.eval()
    predictions_stage0 = []
    probabilities_stage0 = []
    predictions_stage1 = []
    probabilities_stage1 = []
    labels_list = []

    pbar_val = tqdm(enumerate(val_loader_stage0), total=len(val_loader_stage0), desc="[val]", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_stage1(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions_stage0.extend(predicted.cpu().numpy())
            probabilities_stage0.extend(probs.cpu().numpy())
            labels_list.extend(targets.cpu().numpy())

    class0_index = []
    class1_index = []
    for image_num in range(len(predictions_stage0)):
        if (predictions_stage0[image_num] == 0):
            class0_index.append(image_num)
        else:
            class1_index.append(image_num)

    # print(predictions_stage0, labels_list)

    # 先分01和2
    # val_dataset_stage1 = ValDataSet("data/test", val_dataset, class0_index)
    # 先分0和12
    val_dataset_stage1 = ValDataSet("data/test", val_dataset, class1_index)
    val_loader_stage1 = DataLoader(val_dataset_stage1, 
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    pbar_val = tqdm(enumerate(val_loader_stage1), total=len(val_loader_stage1), desc="[val]", leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in pbar_val:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_stage1(inputs)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            predictions_stage1.extend(predicted.cpu().numpy())
            probabilities_stage1.extend(probs.cpu().numpy())

    predictions = []
    probabilities = []

    stage1_image_num = 0
    for image_num in range(len(predictions_stage0)):
        # 先分01和2
        # if predictions_stage0[image_num] == 0:
        #     if predictions_stage1[stage1_image_num] == 0:
        #         predictions.append(0)
        #         prob0 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][0]
        #         prob1 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][1]
        #         prob2 = probabilities_stage0[image_num][1]
        #         probabilities.append([prob0, prob1, prob2])
        #     else:
        #         predictions.append(1)
        #         prob0 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][0]
        #         prob1 = probabilities_stage0[image_num][0] * probabilities_stage1[stage1_image_num][1]
        #         prob2 = probabilities_stage0[image_num][1]
        #         probabilities.append([prob0, prob1, prob2])
        #     stage1_image_num += 1
        # else:
        #     predictions.append(2)
        #     prob0 = probabilities_stage0[image_num][0] / 2
        #     prob1 = probabilities_stage0[image_num][0] / 2
        #     prob2 = probabilities_stage0[image_num][1]
        #     probabilities.append([prob0, prob1, prob2])

        # 先分0和12
        if predictions_stage0[image_num] == 0:
            predictions.append(0)
            prob0 = probabilities_stage0[image_num][0]
            prob1 = probabilities_stage0[image_num][1] / 2
            prob2 = probabilities_stage0[image_num][1] / 2
            probabilities.append([prob0, prob1, prob2])
        else:
            if predictions_stage1[stage1_image_num] == 0:
                predictions.append(1)
                prob0 = probabilities_stage0[image_num][0] 
                prob1 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][0]
                prob2 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][1]
                probabilities.append([prob0, prob1, prob2])
            else:
                predictions.append(2)
                prob0 = probabilities_stage0[image_num][0] 
                prob1 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][0]
                prob2 = probabilities_stage0[image_num][1] * probabilities_stage1[stage1_image_num][1]
                probabilities.append([prob0, prob1, prob2])
            stage1_image_num += 1

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

    val_dataset = DataSet(args.test_image_dir)
    file_names = val_dataset.image_names

    model_stage0_eval = model_dict[args.model]()
    model_stage1_eval = model_dict[args.model]()
    best_model_stage0_path = "saved_models/0/resnet_epoch20.pth"
    best_model_stage1_path = "saved_models/1/resnet_epoch11.pth"

    final_predictions, final_probabilities = evaluate_both_stages(best_model_stage0_path, best_model_stage1_path, val_dataset, model_stage0_eval, model_stage1_eval, device)

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