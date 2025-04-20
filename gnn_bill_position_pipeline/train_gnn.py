# https://github.com/pyg-team/pytorch_geometric/issues/3958

import argparse
import copy
import os
from collections import Counter
from pathlib import Path
from pprint import pprint

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from termcolor import cprint, colored
from torch_geometric.utils import negative_sampling

from model import LobbyHeteroGNN
from preprocess_data import get_input_data
from utils import make_deterministic_everything
from utils_fscache import repr_kv_fn, fscaches, repr_short_kwargs_kv_fn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


TARGET_LINK = ('client', 'lobby', 'bill')
assert TARGET_LINK == LobbyHeteroGNN.target_link()


def get_important_keys(args):
    ik = ["gnn_input", "dataset"] + sorted([
        "model_name",
        "emb_size",
        "num_layers",
        "num_bases",
        "use_skip",
        "use_bn",
        "use_decoder_bn",
        "lr",
    ])
    # if args.label_mapping is not None:
    #     ik = ["label_mapping"] + ik
    # if args.num_epochs is not None:
    #     ik = ik + ["num_epochs"]
    if int(args.nth) != 0:
        assert int(args.nth) == int(args.seed), f"{args.nth} != {args.seed}"
        ik = ik + ["nth"]
    return ik


def get_data_criterion_device(args, load_train_only=False, load_test_only=False):
    device = torch.device("cpu")
    if args.gpu_num >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_num))

    label_mapping = eval(args.label_mapping) if args.label_mapping is not None else None
    train_data, val_data, test_data, _, _ = get_input_data(
        args.gnn_input, args.dataset, label_mapping=label_mapping)

    if load_train_only:
        train_data = train_data.to(device)
        val_data = test_data = None
    elif load_test_only:
        test_data = test_data.to(device)
        train_data = val_data = None
    else:
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)
        print(f"train_labels stats: {Counter(train_data[TARGET_LINK].edge_label.tolist())}")

    _labels = (test_data or train_data)[TARGET_LINK].edge_label
    num_classes = torch.unique(_labels[_labels >= 0]).size(0)
    if args.use_negative_sampling:
        num_classes += 1
    args.num_classes = num_classes

    criterion_train = criterion_test = nn.CrossEntropyLoss()
    # if num_classes == 2:
    #     if args.use_weighted_ce and args.use_negative_sampling:
    #         num_samples = [8688, 38657,
    #                        args.num_negative_samples if args.num_negative_samples is not None else (38657 + 8688)]
    #         weights = torch.FloatTensor([1 - (x / sum(num_samples)) for x in num_samples]).to(device)
    #         criterion_train = nn.CrossEntropyLoss(weight=weights)
    #         criterion_test = nn.CrossEntropyLoss(weight=weights[:2])
    #     elif args.use_weighted_ce and not args.use_negative_sampling:
    #         num_samples = [8688, 38657]
    #         weights = torch.FloatTensor([1 - (x / sum(num_samples)) for x in num_samples]).to(device)
    #         criterion_train = criterion_test = nn.CrossEntropyLoss(weight=weights)

    return train_data, val_data, test_data, criterion_train, criterion_test, device


def get_model_name_path_file_and_data_path(args, important_keys, important_model_keys=None):
    dataset_path = f'{args.gnn_input}/'
    if MODE == "load_best_and_predict":
        dataset_path = 'gnn_output/'
    model_name = repr_kv_fn(*[getattr(args, k) for k in important_keys])
    model_path = f'gnn_output/_checkpoints/{model_name}'
    if important_model_keys is not None:
        model_file = f'{model_path}/model_{repr_kv_fn(*[getattr(args, k) for k in important_model_keys])}.pth'
    else:
        model_file = f'{model_path}/model.pth'
    return model_name, model_path, model_file, dataset_path


def get_model_and_optimizer(args, data, device):
    model = LobbyHeteroGNN(
        data=data,
        layer_name=args.model_name,
        num_layers=args.num_layers,
        hidden_channels=args.emb_size,
        num_bases=args.num_bases,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        activation="elu",
        use_bn=args.use_bn,
        use_skip=args.use_skip,
        dropout_channels=0.2,
        decoder_predictor_type=args.decoder_predictor_type,
        decoder_num_layers=args.decoder_num_layers,
        use_decoder_bn=args.use_decoder_bn,
        logit_slice=None,  # NOTE: important
    ).to(device)
    print(model)
    pprint(args.__dict__)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(data.x_dict, data.edge_index_dict)

    optimizer = model.configure_optimizers()
    return model, optimizer


def train(args, criterion_train, model, train_data, device, optimizer, target_link):
    model.train()
    optimizer.zero_grad()

    target_edge_label = train_data[target_link].edge_label.long()  # Counter({1: 38657, 0: 8688})
    target_edge_label_index = train_data[target_link].edge_label_index

    if args.use_negative_sampling:
        model.logit_slice = None
        neg_edge_index = negative_sampling(
            target_edge_label_index, num_nodes=model.target_sizes(),
            num_neg_samples=args.num_negative_samples,
        )
        neg_edge_labels = torch.full((neg_edge_index.size(1),), fill_value=args.num_classes, device=device)

        target_edge_label_index = torch.cat([target_edge_label_index, neg_edge_index], dim=-1)
        target_edge_label = torch.cat([target_edge_label, neg_edge_labels], dim=-1)

    entity_dict, edge_emb, pred = model(train_data.x_dict, train_data.edge_index_dict,
                                        target_edge_label_index[:, target_edge_label >= 0])

    loss = criterion_train(pred, target_edge_label[target_edge_label >= 0])
    loss.backward(retain_graph=True)  # add retained_graph
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(args, criterion_test, model, data, target_link, datatype):
    model.eval()
    if args.use_negative_sampling:
        model.logit_slice = slice(0, args.num_classes)

    target_edge_label = data[target_link].edge_label.long()  # float()

    target_edge_label_index = data[target_link].edge_label_index[:, target_edge_label >= 0]
    target_edge_label = target_edge_label[target_edge_label >= 0]

    entity_dict, edge_emb, pred = model(data.x_dict, data.edge_index_dict, target_edge_label_index)
    loss = criterion_test(pred, target_edge_label) if criterion_test is not None else None
    prediction = pred.cpu().numpy().argmax(axis=1)
    label = target_edge_label.cpu().numpy()

    acc = accuracy_score(label, prediction) * 100
    macro_f1 = f1_score(label, prediction, average='macro', zero_division=0.0) * 100
    report = classification_report(label, prediction)
    result_dict = classification_report(label, prediction, output_dict=True)
    confusion_matrix_result = confusion_matrix(label, prediction)

    return loss, acc, macro_f1, report, result_dict, confusion_matrix_result, entity_dict, edge_emb, label, prediction


def get_save_file_name(args):
    lst = [args.dataset, args.model_name]
    print(lst)
    # if args.label_mapping is not None:
    #     lst = [f"label_map={args.label_mapping}"] + lst
    return "_".join(lst)


def train_and_test(args, important_keys, important_model_keys=None, save_best_model=True, skip_if_exist=False,
                   data_criterion_device=None, output_log_path="gnn_output/_results", log_kwargs=None):
    model_name, model_path, model_file, dataset_path = get_model_name_path_file_and_data_path(
        args, important_keys, important_model_keys)
    if skip_if_exist and Path(model_file).is_file():
        cprint(f"Skip an existing model: {model_file}", "red")
        return
    elif save_best_model:
        cprint(f"The best model will be saved at {model_file}", "blue")

    save_file_name = get_save_file_name(args)
    csv_file = f'{save_file_name}.csv'
    csv_path_name = os.path.join(output_log_path, csv_file)
    cprint(f"Logs will be saved at: {csv_path_name}", "blue")

    best_val_f1 = final_test_f1 = final_test_f11 = final_test_acc1 = 0
    best_t_edge_emb, best_v_edge_emb, best_test_edge_emb = None, None, None
    best_gt_edge_emb, best_other_edge_emb = None, None
    best_test_label = best_test_prediction = best_test_confusion_m = None
    best_gt_label = best_gt_prediction = None

    if data_criterion_device is None:
        train_data, val_data, test_data, criterion_train, criterion_test, device = get_data_criterion_device(args)
    else:
        train_data, val_data, test_data, criterion_train, criterion_test, device = data_criterion_device
    model, optimizer = get_model_and_optimizer(args, train_data, device)

    for epoch in range(args.num_epochs or 1000):
        loss = train(args, criterion_train, model, train_data, device, optimizer, TARGET_LINK)

        # args, criterion_test, model, data, num_classes, target_link, datatype
        loss, train_acc, train_f1, t_report, t_result_dict, t_confusion_m, t_entity_dict, t_edge_emb, t_label, t_prediction = test(
            args, criterion_train, model, train_data, TARGET_LINK, 'train')
        val_loss, val_acc, val_f1, v_report, v_result_dict, v_confusion_m, v_entity_dict, v_edge_emb, v_label, v_prediction = test(
            args, criterion_train, model, val_data, TARGET_LINK, 'val')
        test_loss1, test_acc1, test_f11, test_report1, test_result_dict, test_confusion_m, test_entity_dict, test_edge_emb, test_label, test_prediction = test(
            args, criterion_train, model, test_data, TARGET_LINK, 'test')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            final_test_acc1 = test_acc1
            final_test_f11 = test_f11

            final_train_acc = train_acc
            final_train_f1 = train_f1
            final_val_acc = val_acc
            final_val_f1 = val_f1
            best_epoch = epoch

            best_report1 = test_report1

            best_t_entity_dict = t_entity_dict
            best_v_entity_dict = v_entity_dict
            best_test_entity_dict = test_entity_dict

            best_t_edge_emb = t_edge_emb
            best_v_edge_emb = v_edge_emb
            best_test_edge_emb = test_edge_emb

            best_test_label = test_label
            best_test_prediction = test_prediction

            best_t_result_dict = t_result_dict
            best_v_result_dict = v_result_dict
            best_test_result_dict = test_result_dict

            best_t_confusion_m = t_confusion_m
            best_v_confusion_m = v_confusion_m
            best_test_confusion_m = test_confusion_m

        if epoch % 20 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, '
                f'\t Train: {train_acc:.2f}, {train_f1:.2f}, '
                f'\t Val: {val_acc:.2f}, {val_f1:.2f}, '
                f'\t Test: {test_acc1:.2f}, {test_f11:.2f}' +
                colored(f"\t Best Test: {final_test_acc1:.2f} {final_test_f11:.2f}", "yellow")
            )

    print()
    print(f'Best epoch: {best_epoch:03d}')
    print(f'model_name: {args.model_name}')
    print("dataset: ", args.dataset)
    print(f'lr: {args.lr}')
    print("emb_size: ", args.emb_size)
    print("nth: ", args.nth, "seed: ", args.seed)
    print(f' Final Train: {best_epoch:03d}  {final_train_acc:.2f} {final_train_f1:.2f}')
    print(f'Final Val: {best_epoch:03d}   {final_val_acc:.2f} {final_val_f1:.2f}')
    print(f'{TARGET_LINK} \n Final Test: {best_epoch:03d}  {final_test_acc1:.2f} {final_test_f11:.2f}')
    print(best_report1)
    print(best_test_confusion_m)

    #### save model ####
    if save_best_model:
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), model_file)
        cprint(f'Saved model at {model_file}', "blue")

    ####################

    #### the lines below are for saving result ###############################################
    result_dict_list = (best_t_result_dict, best_v_result_dict, best_test_result_dict)
    confusion_m_list = (best_t_confusion_m, best_v_confusion_m, best_test_confusion_m)

    meta_info_dict = {'dataset': [args.dataset]}
    if log_kwargs is not None:
        meta_info_dict.update(log_kwargs)
    meta_info_dict.update(dict(model.h))

    class_score_dict = {f"test_f1_{cn}": [best_test_result_dict[str(i)]["f1-score"]] for i, cn in
                        enumerate(train_data[TARGET_LINK].class_names)}
    best_score_dict = {'nth': [args.nth], 'best_epoch': [best_epoch],
                       'train_acc': [final_train_acc], 'train_f1': [final_train_f1],
                       'val_acc': [final_val_acc], 'val_f1': [final_val_f1],
                       'test_acc1': [final_test_acc1], 'test_f11': [final_test_f11],
                       **class_score_dict,
                       "best_report1": [best_report1]}

    best_result_dict = meta_info_dict.copy()
    best_result_dict.update(best_score_dict)

    df = pd.DataFrame(best_result_dict)
    os.makedirs(output_log_path, exist_ok=True)
    if not os.path.isfile(csv_path_name):
        df.to_csv(csv_path_name, index=False)
    else:
        ori_df = pd.read_csv(csv_path_name)
        df = pd.concat([ori_df, df], ignore_index=True)
        df.to_csv(csv_path_name, index=False)

    # save_info_result_dict(output_path, save_file_name, meta_info_dict, result_dict_list, confusion_m_list, args)
    return model, train_data, val_data, test_data, device


@torch.no_grad()
@fscaches(path_attrname_in_kwargs="cache_path", key_fn=repr_short_kwargs_kv_fn,
          keys_to_exclude=["args", "important_keys", "important_model_keys", "model", "device"], verbose=True)
def load_best_and_predict(given_model_name, args, important_keys, important_model_keys, cache_path,
                          model=None, target_data=None, target_attr=None, device=None, **kwargs):
    target_candidates = [None, "train", "val", "test"]
    if target_data in target_candidates and device is None:
        train_data, val_data, test_data, _, _, device = get_data_criterion_device(args)
        target_data = [test_data, train_data, val_data, test_data][target_candidates.index(target_data)]
    if model is None:
        model_name, model_path, model_file, dataset_path = get_model_name_path_file_and_data_path(
            args, important_keys, important_model_keys)
        assert model_name == given_model_name, f"{model_name} != {given_model_name}"
        model, optimizer = get_model_and_optimizer(args, target_data, device)
        cprint(f'Load model from {model_file}', "green")
        model.load_state_dict(torch.load(model_file))
    model.eval()

    test_loss1, test_acc1, test_f11, test_report1, test_result_dict, test_confusion_m, test_entity_dict, test_edge_emb, test_label, test_prediction = test(
        args, None, model, test_data, TARGET_LINK, 'test')
    print("Confusion matrix =")
    print(test_confusion_m)
    print(f"{test_f11=}")
    print("-" * 20)

    if target_attr in [None, "edge_index"]:
        target_edge_index = target_data[TARGET_LINK].edge_index
    elif target_attr in ["edge_label_index"]:
        target_edge_index = target_data[TARGET_LINK].edge_label_index
    else:
        raise ValueError(f"Wrong target_attr: {target_attr}")
    entity_dict, edge_emb, pred = model(target_data.x_dict, target_data.edge_index_dict, target_edge_index)
    print(pred)
    print(pred.shape)
    return pred.cpu()


def get_parser(args_kwargs_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODE', type=str, default='train_and_test')
    parser.add_argument('--label_mapping', type=str, default="[0,1,2,2]")
    parser.add_argument('--gpu_num', type=int, default=7, help='GPU Number.')
    parser.add_argument('--gnn_input', type=str, default='gnn_input')
    parser.add_argument('--dataset', type=str, default="base-")

    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_bases', type=int, default=4, help='the number of bases')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--use_bn', type=str2bool, default=True)
    parser.add_argument('--use_skip', type=str2bool, default=True)
    parser.add_argument('--use_decoder_bn', type=str2bool, default=True)
    parser.add_argument('--nth', type=str, default='0', help='nth')
    parser.add_argument('--model_name', type=str, default='SAGEConv', help='model name')
    parser.add_argument('--emb_size', type=int, default=90, help='embedding size')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--decoder_predictor_type', type=str, default="Concat")
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0, help='seed')

    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--use_weighted_ce', type=str2bool, default=False)
    parser.add_argument('--use_negative_sampling', type=str2bool, default=False)
    parser.add_argument('--num_negative_samples', type=int, default=None)


    if args_kwargs_list is not None:
        for args, kwargs in args_kwargs_list:
            parser.add_argument(args, **kwargs)
    return parser


def build_args_with_the_best(args, metric, output_log_path="gnn_output/_results", args_to_exclude=None):
    args = copy.deepcopy(args)
    csv_path = Path(output_log_path) / f'{get_save_file_name(args)}.csv'
    df = pd.read_csv(csv_path)
    max_row = df.loc[df[metric].idxmax()]

    for k, v in max_row.to_dict().items():
        if hasattr(args, k) and getattr(args, k) != v and k not in args_to_exclude:
            cprint(f"Change args.{k} to '{v}', the best in {csv_path}", "red")
            setattr(args, k, v)
    return args


if __name__ == '__main__':

    __parser__ = get_parser()
    __args__ = __parser__.parse_args()
    make_deterministic_everything(__args__.seed)
    print(f"SEED: {__args__.seed}")

    __important_keys__ = get_important_keys(__args__)
    print(__important_keys__)

    MODE = __args__.MODE
    if MODE == "train_and_test":
        train_and_test(__args__, __important_keys__, save_best_model=True, skip_if_exist=True)

    elif MODE == "load_best_and_predict":
        __args__ = build_args_with_the_best(
            __args__, metric="test_f11",
            args_to_exclude=["seed", "nth"],  # NOTE: important
        )
        _model_name, _, _, _dataset_path = get_model_name_path_file_and_data_path(__args__, __important_keys__)
        for _target_data in ["train", "val", "test"]:
            for _target_attr in ["edge_index", "edge_label_index"]:
                load_best_and_predict(
                    _model_name,
                    args=__args__, important_keys=__important_keys__, important_model_keys=None,
                    target_data=_target_data, target_attr=_target_attr,
                    cache_path=Path(_dataset_path),
                )
