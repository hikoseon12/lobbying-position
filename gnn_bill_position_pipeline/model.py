from pprint import pprint

import lightning as L
import torch
from termcolor import colored, cprint
from torch_geometric.nn import to_hetero_with_bases

from model_utils import GraphEncoder, EdgePredictor


class LobbyEdgeDecoder(torch.nn.Module):
    def __init__(self, data, predictor_type, num_layers, hidden_channels, out_channels,
                 activation="relu", use_bn=False, dropout_channels=0.0):
        super().__init__()
        self.entity1 = data.metadata()[0][0]
        self.entity2 = data.metadata()[0][1]

        self.edge_predictor = EdgePredictor(
            predictor_type,
            num_layers=num_layers, hidden_channels=hidden_channels, out_channels=out_channels,
            activation=activation, use_bn=use_bn, dropout_channels=dropout_channels,
        )
        # self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        # self.lin2 = Linear(hidden_channels, 2)  # for 2 class output

    def forward(self, z_dict, edge_label_index, return_emb=False):
        row, col = edge_label_index
        edge_emb, preds = self.edge_predictor(x_i=z_dict[self.entity1][row],
                                              x_j=z_dict[self.entity2][col],
                                              return_emb=return_emb)
        return edge_emb, preds
        # z = torch.cat([z_dict[self.entity1][row], z_dict[self.entity2][col]], dim=-1)
        # edge_emb = self.lin1(z).relu()
        # z = self.lin2(edge_emb)
        # return edge_emb, z

    def extra_repr(self) -> str:
        return f"[entity]: ({self.entity1}, {self.entity2})"


class LobbyHeteroGNN(L.LightningModule):

    @property
    def h(self):
        return self.hparams

    @staticmethod
    def target_link():
        return "client", "lobby", "bill"

    def target_sizes(self, data=None):
        if data is None:
            return self._target_sizes
        s1 = data.x_dict[self.target_link()[0]].size(0)
        s2 = data.x_dict[self.target_link()[2]].size(0)
        self._target_sizes = (s1, s2)
        return self._target_sizes

    def __init__(self, data,
                 layer_name, num_layers, hidden_channels, num_bases, num_classes, lr,
                 weight_decay=0.0, activation="relu", use_bn=False, use_skip=False, dropout_channels=0.0,
                 decoder_predictor_type="Concat", decoder_num_layers=2, use_decoder_bn=False,
                 logit_slice=None):
        super().__init__()
        self.save_hyperparameters(ignore=["data"])

        self._target_sizes = self.target_sizes(data)
        self.model_name = layer_name
        self.__homo_model_repr__ = None
        if self.model_name == "HAN":
            # self.encoder = HAN(data.metadata(), hidden_channels, hidden_channels)
            raise NotImplementedError
        elif self.model_name == "HGT":
            # self.encoder = HGT(data.metadata(), hidden_channels, hidden_channels)
            raise NotImplementedError
        else:
            # self.encoder = GNNEncoder(self.model_name, hidden_channels, hidden_channels)
            model = GraphEncoder(
                layer_name=layer_name, num_layers=num_layers,
                in_channels=hidden_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                activation=activation, use_bn=use_bn, use_skip=use_skip,
                dropout_channels=dropout_channels, activate_last=True,
                for_hetero=True,  # NOTE: important
            )
            self.__homo_model_repr__ = model.__repr__()
            # self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
            # print("to_hetero_with_bases")
            self.encoder = to_hetero_with_bases(model, data.metadata(), num_bases=num_bases,
                                                in_channels={'x': hidden_channels})
        self.decoder = LobbyEdgeDecoder(
            data, predictor_type=decoder_predictor_type, num_layers=decoder_num_layers,
            hidden_channels=hidden_channels, out_channels=num_classes, activation=activation,
            use_bn=use_decoder_bn, dropout_channels=dropout_channels,
        )

        self.logit_slice = logit_slice

    def extra_repr(self) -> str:
        if self.__homo_model_repr__ is not None:
            return colored(f"[to_hetero_with_bases from]: {self.__homo_model_repr__}", "blue")
        return ""

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        edge_emb, logit = self.decoder(z_dict, edge_label_index, return_emb=True)

        if self.logit_slice is not None:
            logit = logit[:, self.logit_slice]
        return z_dict, edge_emb, logit

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.h.lr, weight_decay=self.h.weight_decay)


if __name__ == '__main__':
    from preprocess_data import get_input_data

    _train_data, _val_data, _test_data, *_ = get_input_data(
        "Client_Bill_Registrant_202457",
        "split-random_feature-all_relation-no_extra_relation"
    )

    _model = LobbyHeteroGNN(
        _train_data,
        lr=0.001, weight_decay=0.001,
        layer_name="GCNConv", num_layers=2, hidden_channels=5, num_bases=3, num_classes=4,
        activation="elu", use_bn=True, use_skip=True,
        dropout_channels=0.2,

        decoder_num_layers=1,
        use_decoder_bn=True,
    )
    print(_model)

    _z_dict, _edge_emb, _logit = _model(_train_data.x_dict,
                                        _train_data.edge_index_dict,
                                        _train_data[LobbyHeteroGNN.target_link()].edge_label_index)
    cprint("- _z_dict", "blue")
    pprint(_z_dict)

    cprint("- _edge_emb", "blue")
    pprint(_edge_emb)

    cprint("- _logit", "blue")
    pprint(_logit)
