import os
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List, Dict, Tuple

import pandas as pd
import torch
from termcolor import cprint
from torch_geometric.data import HeteroData
from torch_geometric.nn import SimpleConv

from utils_basic import open_csv, open_pkl


def get_input_data(data_group_name, dataset_name, load_data_only=True, minimal_schema=True, label_mapping=None):
    input_path = f'{data_group_name}/model_input'
    preprop_path = f'{data_group_name}/preprocessed_dfs'

    cprint(f'Load from {input_path}', "green")

    if data_group_name == "Client_Bill_Registrant_202458":
        dataset_name = f"{dataset_name}_64emb_detailed_bill_state"

    train_data = open_pkl(f'{input_path}/{dataset_name}/train_data.pkl')
    val_data = open_pkl(f'{input_path}/{dataset_name}/val_data.pkl')
    test_data = open_pkl(f'{input_path}/{dataset_name}/test_data.pkl')

    if minimal_schema and dataset_name == "split-random_feature-all_relation-registrant_temp":
        train_data = HeteroData.from_dict({k: v for k, v in train_data.to_dict().items() if "lobbyist" not in k})
        val_data = HeteroData.from_dict({k: v for k, v in val_data.to_dict().items() if "lobbyist" not in k})
        test_data = HeteroData.from_dict({k: v for k, v in test_data.to_dict().items() if "lobbyist" not in k})

    if not load_data_only:
        client_dicts = open_pkl(f'{input_path}/{dataset_name}/client_dicts.pkl')
        bill_dicts = open_pkl(f'{input_path}/{dataset_name}/bill_dicts.pkl')
        legislator_dicts = None
        if os.path.exists(f'{input_path}/{dataset_name}/legislator_dicts.pkl'):
            legislator_dicts = open_pkl(f'{input_path}/{dataset_name}/legislator_dicts.pkl')
        client_df = open_csv(f'{preprop_path}/client_df.csv')
        bill_df = open_csv(f'{preprop_path}/bill_df.csv')
        legislator_df = open_csv(f'{preprop_path}/legislator_df.csv')
    else:
        (client_dicts, bill_dicts, legislator_dicts) = None, None, None
        (client_df, bill_df, legislator_df) = None, None, None

    for data, stage in zip([train_data, val_data, test_data], ["train", "val", "test"]):
        cprint(f"Preprocessing by manual rules at {stage=}", "blue")
        for e, v_dict in data.to_dict().items():
            if "edge_label" in v_dict:
                class_names = ["support", "oppose", "amend", "monitor"]
                if "edge_label_index" not in v_dict:
                    print(f"\t - If no edge_label_index, del edge_label at {e}")
                    del data[e]["edge_label"]
                    continue
                print(f"\t - Align edge_labels with 0 at {e}")
                v_dict["edge_label"] -= v_dict["edge_label"].min()
                C = v_dict["edge_label"].max() + 1
                if C == 5:
                    print(f"\t - if include unlabeled, 'unlabeled' will be -1 at {e}:"
                          f" [0, 1, 2, 3, 4] --> [-1, 0, 1, 2, 3]")
                    v_dict["edge_label"] -= 1
                if label_mapping is not None:
                    mapping = dict(enumerate(label_mapping))
                    print(f"\t - if {label_mapping=}, these classes will be -1 at {e}:"
                          f" {list(range(C - 1))} --> {label_mapping}")
                    merged_class_names = [[] for _ in range(len(set(mapping.values())))]
                    for c, mc in mapping.items():
                        v_dict["edge_label"][v_dict["edge_label"] == c] = mc
                        merged_class_names[mc].append(class_names[c])
                    class_names = ["&".join(ncn) for ncn in merged_class_names]
                data[e]["class_names"] = class_names

    cprint("- train_data", "blue")
    print(train_data)

    cprint("- val_data", "blue")
    print(val_data)

    cprint("- test_data", "blue")
    print(test_data)

    return (
        train_data, val_data, test_data,
        (client_dicts, bill_dicts, legislator_dicts),
        (client_df, bill_df, legislator_df),
    )


class LightFeatures:
    original_feature_metadata: Dict[str, List[Tuple[str, int, str]]] = {
        "client": [
            ("catname", 394, "categorical"),  # 381
            ("state", 54, "categorical"),  # 50

            ("is_gov", 2, "categorical"),
            ("type", 6, "categorical"),
            ("text", 500, "numerical"),
            ("random", 16, "numerical"),
        ],
        "bill": [
            ("subjects_top_term", 34, "categorical"),
            ("final_state", 5, "categorical"),
            ("party", 3, "categorical"),
            ("is_bipartisan", 2, "categorical"),
            ("same_state_client_sponsor", 2, "categorical"),
            ("text", 500, "numerical"),
            ("random", 16, "numerical"),
        ],
        "legislator": [
            ("final_term_type", 2, "categorical"),
            ("party", 3, "categorical"),
            ("state", 56, "categorical"),
            ("gender", 2, "categorical"),
            ("random", 16, "numerical"),
        ],
        "registrant": [
            ("state", 51, "categorical"),  # 48 --> 51
            ("text", 500, "numerical"),
            ("random", 16, "numerical"),
        ],
        "lobbyist": [
            ("ethnicity", 13, "categorical"),
            ("party", 3, "categorical"),
            ("gender", 6, "categorical"),
            ("random", 16, "numerical"),
        ],
    }

    def __init__(self, data=None):
        self.data = data
        self.feature: Dict[str, torch.Tensor] = {}
        self.feature_metadata = defaultdict(list)
        nodes = self.data.metadata()[0]
        for entity, feature_list in self.original_feature_metadata.items():
            if entity in nodes:
                self.feature[entity] = self.data[entity].x
                self.feature_metadata[entity] = deepcopy(feature_list)

    def repr_feature_metadata(self, name):
        metadata = self.feature_metadata[name]
        original_ks = {f for (f, n, t) in self.original_feature_metadata[name]}
        return [((f if f not in original_ks else f"self_{name}_{f}"), n, t) for (f, n, t) in metadata]

    def validate_data(self):
        for k, num_meta_features in self.num_meta_features().items():
            assert num_meta_features == self.feature[k].shape[1], \
                f"{num_meta_features} != {self.feature[k].shape[1]}"

    def link_name(self, s, d):
        nodes, edges = self.data.metadata()
        return [e for e in edges if s == e[0] and d == e[-1]][0]

    def all_sources(self, d):
        nodes, edges = self.data.metadata()
        return [e[0] for e in edges if d == e[-1]]

    def num_meta_features(self):
        return {k: sum([v[1] for v in vs]) for k, vs in self.feature_metadata.items()}

    def aggr_features(self, src: str, dst: str, aggr: str, verbose=False):
        link_name = self.link_name(src, dst)
        edge_index = self.data[link_name].edge_index
        src_x = self.data[src].x
        dst_x = self.data[dst].x
        aggr_x = SimpleConv(aggr=aggr, combine_root=None)((src_x, dst_x), edge_index)
        if verbose:
            print(f"{src=}, {src_x.size()=}")
            print(f"{dst=}, {dst_x.size()=}")
            print(f"{link_name=}")
            print(f"{dst=}, {aggr_x.size()=}")
        return aggr_x

    def add_fix(self, metadata: List[Tuple[str, int]], prefix, postfix):
        return [("_".join([prefix, f, postfix]), n, t) for (f, n, t) in metadata]

    def concat_aggr_all_src_features(self, dst: str, aggr: str, src_entities=None, verbose=True):
        src_list = self.all_sources(dst)
        if src_entities is not None:
            src_list = [s for s in src_list if s in src_entities]
        for src in src_list:
            af = self.aggr_features(src, dst, aggr=aggr)
            self.feature_metadata[dst] += self.add_fix(self.original_feature_metadata[src],
                                                       prefix=f"{aggr}_{src}", postfix=f"of_{dst}")
            self.feature[dst] = torch.cat([self.feature[dst], af], dim=-1)
            if verbose:
                print(f"concat_aggr_all_src_features | {src} --> {dst}: {self.feature[dst].shape}")

    def get_edge_features_dict(self, src, dst):
        link_name = self.link_name(src, dst)
        cprint(f"get_edge_features_dict of {link_name}", "yellow")
        features_dict, meta = {}, None
        for edge_name, edge_tensor in self.data[link_name].items():
            if len(edge_tensor.size()) == 2 and edge_tensor.size(0) == 2:
                src_x, dst_x = self.feature[src], self.feature[dst]
                src_e, dst_e = src_x[edge_tensor[0]], dst_x[edge_tensor[1]]
                edge_features = torch.cat([src_e, dst_e], dim=-1)
                features_dict[edge_name] = edge_features
                meta = self.repr_feature_metadata(src) + self.repr_feature_metadata(dst)
        return features_dict, meta

    def get_E_and_y_and_meta(self, src, dst, edge_name="edge_label_index", y_name="edge_label"):
        edge_features_dict, meta = self.get_edge_features_dict(src, dst)
        E = edge_features_dict[edge_name]
        y = self.data[self.link_name(src, dst)].get(y_name)
        return E, y, meta

    def postprocess(self, feature_tensor, name_length_dtype_list, out_type="df"):
        ptr = 0
        sparse_feature_list, cols = [], []
        for feature, length, dtype in name_length_dtype_list:
            if dtype == "categorical" and length > 1:
                # categorical one-hot to sparse integer
                print(feature, length, dtype)
                sub_feat = feature_tensor[:, ptr:ptr + length]

                if feature.startswith("self"):
                    # Only one variable is allocated for one-hot encoding
                    aggr_x_to_validate = torch.sum(sub_feat, dim=-1)
                    if sub_feat.size(0) != torch.sum(aggr_x_to_validate == 1).item():
                        raise ValueError(
                            f"mismatch ({feature}): {sub_feat.size(0)} != {torch.sum(aggr_x_to_validate == 1).item()}")
                    non_zero_indices = torch.nonzero(sub_feat)
                    sparse_feature_list.append(non_zero_indices[:, 1].view(-1, 1))
                    cols += [feature]
                else:  # aggregated features (mean, sum, ...)
                    sparse_feature_list.append(sub_feat)
                    cols += [f"{feature}_{i}" for i in range(length)]
            ptr += length

        sparse_feature = torch.cat(sparse_feature_list, dim=-1)
        if out_type == "df":
            df = pd.DataFrame(sparse_feature.numpy(), columns=cols)
            return df
        else:
            return sparse_feature
