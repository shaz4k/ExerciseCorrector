import pickle
import numpy as np
import random
import sys
import torch
import pandas as pd
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
from tabulate import tabulate

labels_dict = {0: ('Squat', 'Correct'), 1: ('Squat', 'Feet too wide'), 2: ('Squat', 'Knees inward'),
               3: ('Squat', 'Not low enough'), 4: ('Squat', 'Leaning forward'),
               6: ('Lunges', 'Correct'), 7: ('Lunges', 'Not low enough'), 8: ('Lunges', 'Knees pass toe'),
               9: ('Plank', 'Correct'), 10: ('Plank', 'Arched back'), 11: ('Plank', 'Hunch back')}
# correct = 0, 6, 9

labels_dict2 = {0: 'Squat - Correct', 1: 'Squat - Wide Feet', 2: 'Squat - Knees In',
                3: 'Squat - Not Low', 4: 'Squat - Leaning Fwd',
                6: 'Lunges - Correct', 7: 'Lunges - Not Low', 8: 'Lunges - Knees Pass Toe',
                9: 'Plank - Correct', 10: 'Plank - Arched Back', 11: 'Plank - Hunch Back'}


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

    # For display
    classes2 = list(labels_dict2.values())
    cm2 = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes2, yticklabels=classes2,
                annot_kws={"size": 14}, cbar=False, square=True, linewidths=0.5)
    # Add vertical lines
    plt.axvline(5, color='black', linewidth=2, ymin=-0.1, ymax=1.1)  # between Squat and Lunges
    plt.axvline(8, color='black', linewidth=2, ymin=-0.1, ymax=1.1)  # between Lunges and Plank

    # Add horizontal lines
    plt.axhline(5, color='black', linewidth=2, xmin=-0.1, xmax=1.1)  # between Squat and Lunges
    plt.axhline(8, color='black', linewidth=2, xmin=-0.1, xmax=1.1)  # between Lunges and Plank

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    print(f'Accuracy Score: {acc * 100} %')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}, cbar=False, square=True, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy_dict = {
        label: accuracy
        for label, accuracy in zip(labels_dict.values(), per_class_acc * 100)
    }
    # Print per-class accuracy in a table
    table_data = [
        (label, f"{accuracy:.2f} %") for label, accuracy in per_class_accuracy_dict.items()
    ]
    print(tabulate(table_data, headers=["Label", "Accuracy"], tablefmt="pretty"))

    # Original results
    data = {
        ('Squat', 'Correct'): 90.0,
        ('Squat', 'Feet too wide'): 100,
        ('Squat', 'Knees inward'): 100,
        ('Squat', 'Not low enough'): 100,
        ('Squat', 'Leaning forward'): 57.1,
        ('Lunges', 'Correct'): 66.7,
        ('Lunges', 'Not low enough'): 100,
        ('Lunges', 'Knees pass toe'): 100,
        ('Plank', 'Correct'): 85.7,
        ('Plank', 'Arched back'): 100,
        ('Plank', 'Hunch back'): 100
    }

    # Convert the given data into a pandas DataFrame
    data_df = pd.DataFrame({'Label': list(data.keys()), 'Original (GCN)': list(data.values())})

    # Existing code (assuming you have cm and labels_dict)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy_dict = {
        label: accuracy
        for label, accuracy in zip(labels_dict.values(), per_class_acc * 100)
    }

    labels, accuracies = zip(*per_class_accuracy_dict.items())
    cm_df = pd.DataFrame({'Label': labels, 'New (CNN)': accuracies})

    # Merge data and cm
    merged_df = pd.merge(data_df, cm_df, on='Label')
    merged_df = merged_df.melt(id_vars='Label', var_name='Source', value_name='Accuracy')
    print("Merged DataFrame:")
    print(merged_df)

    # Update the 'Label' column to only include the description
    # merged_df['Label'] = merged_df['Label'].apply(lambda x: x[1])
    merged_df['Tuple_Label'] = merged_df['Label'].apply(lambda x: (x[0], x[1]))

    exercises = ['Squat', 'Lunges', 'Plank']

    for exercise in exercises:
        exercise_keys = [k for k, v in labels_dict.items() if v[0] == exercise]

        exercise_df = merged_df[merged_df['Tuple_Label'].isin([(exercise, labels_dict[k][1]) for k in exercise_keys])]
        exercise_df['Label'] = exercise_df['Label'].apply(lambda x: x[1])

        # exercise_df = merged_df[merged_df['Label'].isin([labels_dict[k][1] for k in exercise_keys])]
        print(f"\n{exercise} DataFrame:")
        print(exercise_df)

        if exercise == 'Squat':
            custom_palette = sns.color_palette(["#D4A0A7", "#A22522"])  # reds
        if exercise == 'Lunges':
            custom_palette = sns.color_palette(["#FFC971", "#CC5803"])  # oranges
        if exercise == 'Plank':
            custom_palette = sns.color_palette(["#A1B5D8", "#355070"])  # blues

        g = sns.catplot(data=exercise_df, x='Label', y='Accuracy', hue='Source', kind='bar', height=5, aspect=1,
                        palette=custom_palette, errorbar=None, width=0.5)
        g.fig.suptitle(f"Comparison of Classification Techniques: {exercise}", fontsize=16)
        g.set_xticklabels(rotation=30, ha='right', fontsize=12)
        g.set_ylabels('Accuracy (%)', fontsize=14)
        g.set_xlabels('')
        g.set(ylim=(0, 110))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for ax in g.axes.flat:
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                            ha='center', va='bottom', fontsize=10)

        sns.move_legend(g, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True, facecolor='white', fontsize=12,
                        title='Model')
        plt.tight_layout()
    plt.show()

    # # # Filter out rows within a label where both have 100% accuracy
    # # mask = merged_df.groupby('Label')['Accuracy'].transform(lambda x: not (x == 100).all())
    # # merged_df = merged_df[mask]
    #
    # # Plot the updated DataFrame using catplot
    # custom_palette = sns.color_palette(["#4C72B0", "#EB9C5C"])  # blue and orange
    #
    # g = sns.catplot(data=merged_df, x='Label', y='Accuracy', hue='Source', kind='bar', height=6, aspect=2, palette=custom_palette)
    # g.fig.suptitle("Comparison of Classification Techniques", fontsize=16)
    # g.set_xticklabels(rotation=30, ha='right', fontsize=12)
    # g.set_ylabels('Accuracy (%)', fontsize=14)
    # g.set_xlabels('')
    # g.set(ylim=(0, 110))
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    #
    # for ax in g.axes.flat:
    #     for p in ax.patches:
    #         # if p.get_height() < 100:  # Only add text for values less than 100%
    #         ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
    #                     ha='center', va='bottom', fontsize=10)
    #
    # sns.move_legend(g, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True, facecolor='white', fontsize=12, title='Model')
    # plt.tight_layout()
    # plt.show()


def cnn_checker(class_path, corr_path, corr_model):
    data_train, data_test = load_DCT()

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
            if input_label != pred_label:
                viz = SkeletonVisualizer(num_frames=pred.shape[0], data1=model_in, data2=pred, add_labels=True,
                                         translate_to_joint=None,
                                         text=f'Input Label: {gt_labels[i]} - {input_label}\n'
                                              f'Output Label: {predicted[i]} - {pred_label}')
            # if input_label[0] == 'Lunges':
            #     viz.plot_still(50)
            # else:
            #     viz.plot_still(20)
            # plt.close()
                viz.show(save_filename=f'results/{i}.gif')
                plt.close()

    # Calculate Accuracy
    correct_lab = ([0, 6, 9])

    # Initialize dictionaries to keep track of the corrected count and total count for each label
    correction_count = {k: 0 for k in labels_dict.keys()}
    label_count = {k: 0 for k in labels_dict.keys()}

    total_incorrect = 0
    corrected = 0
    for gt, pred in zip(gt_labels, predicted):
        label_count[gt] += 1
        if gt not in correct_lab:
            total_incorrect += 1
            if pred in correct_lab:
                corrected += 1
                correction_count[gt] += 1
        elif gt in correct_lab and gt == pred:
            correction_count[gt] += 1

    # Calculate the percentage of corrected labels for each type
    percentage_corrected = round((corrected / total_incorrect) * 100)
    percentage_corrected_per_label = {k: round((v / label_count[k]) * 100) for k, v in correction_count.items()}

    # Print results
    print("Total percentage corrected:", percentage_corrected)
    print("Percentage corrected per label:")
    for k, v in percentage_corrected_per_label.items():
        print(f"{labels_dict[k][0]} ({labels_dict[k][1]}): {v}%")

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
    corrected = round((correct / total) * 100)

    print(f'Total percentage correct: {corrected}%')

    classes = [val for key, val in labels_dict2.items()]

    cm = confusion_matrix(predicted, gt_labels)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14}, cbar=False, square=True, linewidths=0.5)
    # Add vertical lines
    plt.axvline(5, color='black', linewidth=2, ymin=-0.1, ymax=1.1)  # between Squat and Lunges
    plt.axvline(8, color='black', linewidth=2, ymin=-0.1, ymax=1.1)  # between Lunges and Plank

    # Add horizontal lines
    plt.axhline(5, color='black', linewidth=2, xmin=-0.1, xmax=1.1)  # between Squat and Lunges
    plt.axhline(8, color='black', linewidth=2, xmin=-0.1, xmax=1.1)  # between Lunges and Plank

    plt.xlabel('Input Label')
    plt.ylabel('Corrected Label')
    plt.tight_layout()
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
    available_eval = ['(1) Classifier', '(2) Improved GCN Corrector',
                      '(3) GCN with Attention']
    cnn_dir = 'runs/3D-CNN-Classifier/checkpoints/2023-04-08_19-15-09.pt'
    gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-10_22-41-08.pt'
    gcn_attn_dir = 'runs/GCN_Corrector_Attn/checkpoints/2023-04-17_16-50-07.pt'
    # gcn_corr_eval(gcn_dir)
    # visualise_weights(cnn_dir)
    while True:
        select = int(input(f'Please enter number for the evaluation you want to perform\n{available_eval}'))
        if select == 1:
            class_eval(cnn_dir)
            sys.exit()
        if select == 2:
            corr_model = GCN_Corrector()
            gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-19_22-53-58.pt'
            # gcn_dir = 'runs/GCN_Corrector/checkpoints/2023-04-17_17-11-08.pt'
            cnn_checker(cnn_dir, gcn_dir, corr_model)
            sys.exit()
        if select == 3:
            corr_model = GCN_Corrector_Attention()
            gcn_attn_dir = 'runs/GCN_Corrector_Attn/checkpoints/2023-04-19_21-46-55.pt'
            cnn_checker(cnn_dir, gcn_attn_dir, corr_model)
            sys.exit()
        else:
            print('Please input valid number!')
