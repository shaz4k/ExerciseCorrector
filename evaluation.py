import pickle
import numpy as np
import random
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.train_utils import get_labels
from models import CNN_Classifier, GCN_Corrector, GCN_Corrector_Attention
from sklearn.metrics import accuracy_score
from visualisation import SkeletonVisualizer
import matplotlib.pyplot as plt
from utils.train_utils import load_DCT

labels_dict = {0: ('SQUAT', 'Correct'), 1: ('SQUAT', 'Feets too wide'), 2: ('SQUAT', 'Knees inward'),
               3: ('SQUAT', 'Not low enough'), 4: ('SQUAT', 'Front bended'),
               6: ('Lunges', 'Correct'), 7: ('Lunges', 'Not low enough'), 8: ('Lunges', 'Knees pass toes'),
               9: ('Plank', 'Correct'), 10: ('Plank', 'Arched back'), 11: ('Plank', 'Rolled back')}
# correct = 0, 6, 9


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    args = checkpoint['args']
    results = checkpoint['results']

    return model, args, results


def class_eval(path):
    temp_path = 'data/EC3D/tmp_3d.pickle'
    try:
        print('Loading saved data.')
        with open(temp_path, "rb") as f:
            data = pickle.load(f)
        data_test = data['test']
    except FileNotFoundError:
        print('File Not Found')
        sys.exit()

    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    model = CNN_Classifier(in_channels=3)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        model.cuda()

    # load weights
    # path = 'runs/3D-CNN-Classifier/checkpoints/2023-03-17_01-24-56_epoch39.pt'
    model, args, results = load_checkpoint(model, path)
    # Set the model to evaluation mode
    model.eval()

    y_true = []
    y_pred = []
    # classes = range(11)
    classes = [val for key, val in labels_dict.items()]

    for i, (batch_id, inputs) in enumerate(test_loader):
        if inputs.shape[1] > 3:
            inputs = inputs.permute(0, 3, 2, 1)
        if is_cuda:
            inputs = inputs.float().cuda()
        else:
            inputs = inputs.float()

        labels = get_labels([test_loader.dataset.inputs_label[int(i)] for i in batch_id], level=1).cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

    acc = accuracy_score(y_true, y_pred)

    print(f'Accuracy Score: {acc * 100} %')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}, cbar=False, square=True, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def cnn_checker(class_path, corr_path, corr_model):
    data_train, data_test = load_DCT()
    # temp_path = 'data/EC3D/tmp_wo_val.pickle'
    # temp_path = 'data/EC3D/tmp_DCT_1CH.pickle'
    # try:
    #     print('Loading saved data.')
    #     with open(temp_path, "rb") as f:
    #         data = pickle.load(f)
    #     data_test = data['test']
    # except FileNotFoundError:
    #     print('File Not Found')
    class_model = CNN_Classifier(in_channels=3)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        class_model.cuda()

    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test), shuffle=False)

    # Load CNN
    class_model, _, _ = load_checkpoint(class_model, class_path)
    class_model.eval()

    # Load GCN results
    # corr_model = GCN_Corrector()
    # corr_model = Simple_GCN_Attention()
    _, _, results = load_checkpoint(corr_model, corr_path)

    inputs = results['in']
    preds = results['out']
    targets = results['targ']
    gt_labels = results['labels']

    # Reshape into [frames, joints, 3] then normalise and Resample
    inputs = [sample.T.reshape(-1, 19, 3) for sample in inputs]
    preds = [sample.T.reshape(-1, 19, 3) for sample in preds]

    # Convert preds to tensor
    input_preds = process_data(preds, target_frames=24, normalization='sample')
    input_preds = process_data(preds, target_frames=24, normalization='sample')
    input_preds = [torch.from_numpy(pred) for pred in input_preds]
    input_preds = torch.stack(input_preds, dim=0)
    input_preds = torch.permute(input_preds, (0, 3, 2, 1))

    if is_cuda:
        input_preds = input_preds.float().cuda()
    else:
        input_preds = input_preds.float()

    # Forward
    outputs = class_model(input_preds)
    _, predicted = torch.max(outputs.data, 1)

    # Normalise data for viz
    # inputs = inputs.
    gt_labels = np.array(gt_labels)
    predicted = predicted.cpu().detach().numpy()

    # Visualise output pred and labels
    show_plots = input('Do you want to show plots? (y/n)\n')
    if show_plots == 'y':
        for i, pred in enumerate(preds):
            input_label = labels_dict[gt_labels[i]]
            pred_label = labels_dict[predicted[i]]
            model_in = inputs[i]
            viz = SkeletonVisualizer(num_frames=pred.shape[0], data1=model_in, data2=pred, add_labels=True, translate_to_joint=18,
                                     text=f'Input Label: {gt_labels[i]} - {input_label}\n'
                                          f'Correction Label: {predicted[i]} - {pred_label}')
            viz.plot_still(20)
            # viz.show()

    # Calculate Accuracy
    correct_lab = ([0, 6, 9])
    total_incorrect = 0
    corrected = 0
    for gt, pred in zip(gt_labels, predicted):
        if gt not in correct_lab:
            total_incorrect += 1
            if pred in correct_lab:
                corrected += 1

    # Calculate the percentage of corrected labels
    percentage_corrected = round((corrected / total_incorrect) * 100)

    print(f"Percentage of corrected labels: {percentage_corrected}%")

    # Total Correct
    correct = 0
    for pred in predicted:
        if pred in correct_lab:
            correct += 1
    total = len(gt_labels)
    corrected = round((correct/total) * 100)

    print(f'Total percentage correct: {corrected}%')

    classes = [val for key, val in labels_dict.items()]

    cm = confusion_matrix(predicted, gt_labels)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}, cbar=False, square=True, linewidths=0.5)
    plt.xlabel('Input Label')
    plt.ylabel('Corrected Label')
    plt.show()



def process_data(data, target_frames=None, normalization=None):
    processed_list = []

    if normalization == 'dataset':
        max_val = np.max([np.max(sample) for sample in data])
        min_val = np.min([np.min(sample) for sample in data])

    for sample in data:
        if normalization == 'sample':
            max_val = np.max(sample)
            min_val = np.min(sample)

        if normalization in ['dataset', 'sample']:
            sample = (2.0 * (sample - min_val) / (max_val - min_val)) - 1.0

        if target_frames:
            num_frames = sample.shape[0]
            if num_frames > target_frames:
                step_size = num_frames // target_frames
                sample = sample[range(0, num_frames, step_size)][:target_frames]

        processed_list.append(sample)

    return processed_list


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    available_eval = ['(1) Classifier', '(2) Original GCN Corrector', '(3) Improved GCN Corrector', '(4) GCN with Attention']
    cnn_dir = 'runs/3D-CNN-Classifier/checkpoints/2023-04-08_19-15-09.pt'
    gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-10_22-41-08.pt'
    gcn_attn_dir = 'runs/GCN_Corrector_Attn/checkpoints/2023-04-17_16-50-07.pt'
    # gcn_corr_eval(gcn_dir)
    # visualise_weights(cnn_dir)
    while True:
        select = int(input(f'Please enter number for the evaluation you want to perform\n{available_eval}'))
        if select == 1:
            class_eval(cnn_dir)
        if select == 2:
            pass
        if select == 3:
            corr_model = GCN_Corrector()
            gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-19_22-53-58.pt'
            # gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-17_17-11-08.pt'
            cnn_checker(cnn_dir, gcn_dir, corr_model)
            sys.exit()
        if select == 4:
            corr_model = GCN_Corrector_Attention()
            gcn_attn_dir = 'runs/GCN_Corrector_Attn/checkpoints/2023-04-19_21-46-55.pt'
            cnn_checker(cnn_dir, gcn_attn_dir, corr_model)
            sys.exit()
        else:
            print('Please input valid number!')




