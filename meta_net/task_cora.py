import random
import numpy as np


def multitask_data_generator(labels, labeled_node_list, select_array, k_spt, k_val, k_qry, n_way):
    labels_local = labels  # .clone().detach()
    # random.shuffle(select_array_index)

    class_idx_list = []
    train_class_list = []
    val_class_list = []
    test_class_list = []
    for i in range(len(select_array)):
        class_idx_list.append([])
        train_class_list.append([])
        val_class_list.append([])
        test_class_list.append([])

    for j in labeled_node_list:
        for i in range(len(select_array)):
            if (labels_local[j] == select_array[i]):
                class_idx_list[i].append(j)

    usable_labels = []
    for i in range(len(class_idx_list)):
        if len(class_idx_list[i]) > 10:
        # if len(class_idx_list[i]) >= 100:
            usable_labels.append(i)
        # elif 15 > len(class_idx_list[i]) > 0:
        #     new_classes.append(i)

    len_usable_labels = len(usable_labels)
    print('len_usable_labels', len_usable_labels)
    random.shuffle(usable_labels)

    base_classes = np.random.choice(usable_labels, len_usable_labels//2, replace=False).tolist()
    new_classes = list(set(usable_labels)-set(base_classes))

    for i in range(len(select_array)):
        if i not in set(usable_labels):
            continue
        # train_class_list[i] = random.sample(class_idx_list[i], k_spt)
        train_class_list[i] = np.random.choice(class_idx_list[i], k_spt, replace=False).tolist()
        val_class_temp = [n1 for n1 in class_idx_list[i] if n1 not in train_class_list[i]]
        # print('val_class_temp', val_class_temp)
        # test_class_list[i] = random.sample(test_class_list[i], k_qry)
        val_class_list[i] = np.random.choice(val_class_temp, k_val, replace=False).tolist()
        # test_class_temp = [n1 for n1 in class_idx_list[i] if
        #                    (n1 not in train_class_list[i]) and (n1 not in val_class_list[i])]
        test_class_temp = [n1 for n1 in class_idx_list[i] if
                           (n1 not in train_class_list[i]) and (n1 not in val_class_list[i])]
        # test_class_list[i] = [np.random.choice(test_class_temp, replace=False).item() for _ in range(k_qry)]
        test_class_list[i] = test_class_temp

    train_idx = [[], []]
    test_idx = [[], []]
    val_idx = [[], []]

    for j in base_classes:
        train_idx[0] += train_class_list[j]
        val_idx[0] += val_class_list[j]
        test_idx[0] += test_class_list[j]

    for j in new_classes:
        train_idx[1] += train_class_list[j]
        val_idx[1] += val_class_list[j]
        test_idx[1] += test_class_list[j]

    task_list = [base_classes, new_classes]

    return task_list, train_idx, val_idx, test_idx
