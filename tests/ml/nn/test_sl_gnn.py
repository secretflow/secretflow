import dgl
import numpy as np
import tensorflow as tf
from dgl.nn.tensorflow import GraphConv
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from secretflow.ml.nn import SLModel
from secretflow.ml.nn.sl.graph_utils import NodeDataLoader
from secretflow.utils.simulation.datasets import load_cora
from secretflow.utils.simulation.tf_gnn_model import GraphAttention, ServerNet
from tests.basecase import DeviceTestCase

hidden_size = 256
n_classes = 7
attn_heads = 2
layer_num = 3
learning_rate = 1e-3
dropout_rate = 0.0
l2_reg = 0.1
num_heads = 4
epochs = 10
optimizer = 'adam'


def create_dataset_builder(
    stage="train",
    train_ratio=0.5,
    n_layers=2,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    seed=321,
):
    def dataset_builder(x):
        assert len(x) == 2, f"input should be list of [nodes, edges]"
        papers, citations = x[0], x[1]

        # convert paper ids into zero-based indices.
        paper_idx = {
            name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))
        }
        papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
        citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
        citations["target"] = citations["target"].apply(lambda name: paper_idx[name])

        # convert subjects into zero-based indices.
        if papers.columns[-1] == "subject":
            class_values = sorted(papers["subject"].unique())
            class_idx = {name: id for id, name in enumerate(class_values)}
            papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

        # create graph
        g = dgl.graph((citations["source"], citations["target"]), idtype=tf.int32)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        if papers.columns[-1] == "subject":
            g.ndata["feat"] = tf.convert_to_tensor(
                papers[papers.columns[1:-1]].values, dtype=tf.float32
            )
            g.ndata["label"] = tf.convert_to_tensor(
                papers["subject"].values, dtype=tf.float32
            )
        else:
            g.ndata["feat"] = tf.convert_to_tensor(
                papers[papers.columns[1:]].values, dtype=tf.float32
            )

        # split train and test dataset
        rs = np.random.RandomState(seed=seed)
        mask = rs.rand(len(g.nodes())) < train_ratio
        if stage == "train":
            indices = g.nodes()[mask]
        else:
            indices = g.nodes()[~mask]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

        return NodeDataLoader(
            g,
            indices,
            sampler,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )

    return dataset_builder


class StochasticTwoLayerGCN(tf.keras.Model):
    def __init__(
        self,
        in_feats,
        n_hidden,
    ):
        super(StochasticTwoLayerGCN, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.conv1 = GraphConv(in_feats, n_hidden)
        self.conv2 = GraphConv(n_hidden, n_hidden)

    def call(self, inputs):
        # flake8: noqa
        # g, input_nodes, output_nodes, blocks = (
        _, _, _, blocks = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )
        input_features = blocks[0].srcdata["feat"]
        x = self.conv1(blocks[0], input_features)
        x = self.conv2(blocks[1], x)

        # NOTE: we should return labels here since NodeDataLoader can't determine labels ahead.
        if "label" in blocks[-1].dstdata:
            return blocks[-1].dstdata["label"], x

        return x

    def get_config(self):
        return {"in_feats": self.in_feats, "n_hidden": self.n_hidden}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GCNLinkPrediction(tf.keras.Model):
    def __init__(self, in_feats, n_hidden, seed):
        super(GCNLinkPrediction, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.conv1 = GraphConv(in_feats, n_hidden)
        self.conv2 = GraphConv(n_hidden, n_hidden)
        self.seed = seed
        self.random_state = np.random.RandomState(seed=seed)

    def call(self, inputs):
        # g, input_nodes, output_nodes, blocks = (
        _, _, output_nodes, blocks = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )

        input_features = blocks[0].srcdata["feat"]
        x = self.conv1(blocks[0], input_features)
        x = self.conv2(blocks[1], x)

        # positive samples
        edges = blocks[-1].edges()
        mask = edges[0] < len(output_nodes)
        pos_src, pos_dst = edges[0][mask], edges[1][mask]

        # negative samples
        # step 1. generate edges whose source and destination nodes are in output_nodes
        n = len(output_nodes)
        neg_src, neg_dst = self.random_state.choice(n, 2 * n), self.random_state.choice(
            n, 2 * n
        )
        # step 2. exclude edges which are positive samples
        mask = blocks[-1].has_edges_between(neg_src, neg_dst)
        neg_src, neg_dst = neg_src[~mask], neg_dst[~mask]
        # step 3. random choice same number of negative samples as positive samples
        mask = self.random_state.choice(len(neg_src), len(pos_src), replace=False)
        neg_src, neg_dst = neg_src[mask], neg_dst[mask]

        # combine positive samples and negative samples
        src = tf.concat((pos_src, neg_src), axis=0)
        dst = tf.concat((pos_dst, neg_dst), axis=0)
        labels = tf.concat((tf.ones(len(pos_src)), tf.zeros(len(neg_src))), axis=0)

        # NOTE: we should return labels here since NodeDataLoader can't determine labels ahead.
        if "label" in blocks[-1].dstdata:
            return labels, x, {"edges": (src, dst)}

        return x

    def get_config(self):
        return {"in_feats": self.in_feats, "n_hidden": self.n_hidden, "seed": self.seed}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FuseLinkPrediction(tf.keras.Model):
    def __init__(self):
        super(FuseLinkPrediction, self).__init__()
        self.dense1 = layers.Dense(16)
        self.dense2 = layers.Dense(1, name="logits")

    def call(self, hiddens, edges):
        x = tf.concat(hiddens, axis=-1)
        src, dst = edges[0], edges[1]
        src_representations = tf.gather(x, src)
        dst_representations = tf.gather(x, dst)
        edge_representations = tf.concat(
            (src_representations, dst_representations), axis=-1
        )

        x = self.dense1(edge_representations)
        return self.dense2(x)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# base model for node classification
def create_base_model(input_shape, n_hidden):
    def base_model():
        feature_input = tf.keras.Input(shape=(input_shape[1],))
        graph_input = tf.keras.Input(shape=(input_shape[0],))
        regular = tf.keras.regularizers.l2(l2_reg)
        outputs = GraphAttention(
            F_=n_hidden,
            attn_heads=num_heads,
            attn_heads_reduction='average',  # {'concat', 'average'}
            dropout_rate=dropout_rate,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            attn_kernel_initializer='glorot_uniform',
            kernel_regularizer=regular,
            bias_regularizer=None,
            attn_kernel_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            attn_kernel_constraint=None,
        )([feature_input, graph_input])
        # outputs = tf.keras.layers.Flatten()(outputs)
        model = Model(inputs=[feature_input, graph_input], outputs=outputs)
        model._name = "embed_model"
        # Compile model
        model.summary()
        metrics = ['acc']
        optimizer = tf.keras.optimizers.get(
            {
                'class_name': 'adam',
                'config': {'learning_rate': learning_rate},
            }
        )
        model.compile(
            loss='categorical_crossentropy',
            weighted_metrics=metrics,
            optimizer=optimizer,
        )
        return model

    return base_model


# fuse model for node classification
def create_fuse_model(hidden_units, n_classes):
    def fuse_model():
        inputs = [keras.Input(shape=size) for size in hidden_units]
        x = layers.concatenate(inputs)
        input_shape = x.shape[-1]
        y_pred = ServerNet(
            in_channel=input_shape,
            hidden_size=hidden_size,
            num_layer=layer_num,
            num_class=n_classes,
            dropout=0.0,
        )(x)
        # Create the model.
        model = keras.Model(inputs=inputs, outputs=y_pred, name="fuse_model")
        model.summary()
        metrics = ['acc']
        optimizer = tf.keras.optimizers.get(
            {
                'class_name': 'adam',
                'config': {'learning_rate': learning_rate},
            }
        )
        model.compile(
            loss='categorical_crossentropy',
            weighted_metrics=metrics,
            optimizer=optimizer,
        )
        return model

    return fuse_model


# base model for link prediction
def create_base_model_link_prediction(in_feats, n_hidden, learning_rate=0.01, seed=123):
    def base_model():
        model = GCNLinkPrediction(in_feats, n_hidden, seed)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )
        return model

    return base_model


# fuse model for link prediction
def create_fuse_model_link_prediction(learning_rate=0.01):
    def fuse_model():
        model = FuseLinkPrediction()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(name="acc")],
        )
        return model

    return fuse_model


class TestSLGraphModel(DeviceTestCase):
    def test_node_classification(self):
        (
            graph,
            features,
            train_y,
            valid_y,
            test_y,
            idx_train,
            idx_val,
            idx_test,
        ) = load_cora([self.alice, self.bob])
        partition_shapes = features.partition_shape()

        input_shape_alice = partition_shapes[self.alice]
        input_shape_bob = partition_shapes[self.bob]

        sl_model = SLModel(
            base_model_dict={
                self.alice: create_base_model(input_shape_alice, hidden_size),
                self.bob: create_base_model(input_shape_bob, hidden_size),
            },
            device_y=self.alice,
            model_fuse=create_fuse_model([hidden_size, hidden_size], n_classes),
        )

        history = sl_model.fit(
            x=[features, graph],
            y=train_y,
            epochs=epochs,
            batch_size=input_shape_alice[0],
            sample_weight=idx_train,
            validation_data=([features, graph], valid_y, idx_val),
        )
        print(history)
        # self.assertGreater(history["train_acc"][-1], 0.75)
        sl_model.save_model(
            base_model_path={
                self.alice: "base_alice",
                self.bob: "base_bob",
            },
            fuse_model_path="fuse_model",
            save_traces=False,
        )

        metrics = sl_model.evaluate(
            x=[features, graph],
            y=test_y,
            batch_size=input_shape_alice[0],
            sample_weight=idx_test,
        )
        print(metrics)

        sl_model_ = SLModel(
            base_model_dict={
                self.alice: None,
                self.bob: None,
            },
            device_y=self.alice,
            model_fuse=None,
        )
        sl_model_.load_model(
            base_model_path={
                self.alice: "base_alice",
                self.bob: "base_bob",
            },
            fuse_model_path="fuse_model",
            base_custom_objects={'GraphAttention': GraphAttention},
            fuse_custom_objects={'ServerNet': ServerNet},
        )
        metrics_ = sl_model_.evaluate(
            x=[features, graph],
            y=test_y,
            batch_size=input_shape_alice[0],
            sample_weight=idx_test,
        )
        self.assertAlmostEqual(metrics['acc'], metrics_['acc'], 2)
        self.assertAlmostEqual(metrics['loss'], metrics_['loss'], 2)

    def link_prediction(self):
        nodes, edges = load_cora([self.alice, self.bob])
        partition_columns = nodes.partition_columns

        n_hidden = 8
        learning_rate = 0.01
        num_epochs = 30
        batch_size = 256

        in_feats_alice = len(partition_columns[self.alice]) - 1
        in_feats_bob = len(partition_columns[self.bob]) - 2

        sl_model = SLModel(
            base_model_dict={
                self.alice: create_base_model_link_prediction(
                    in_feats_alice, n_hidden, learning_rate
                ),
                self.bob: create_base_model_link_prediction(
                    in_feats_bob, n_hidden, learning_rate
                ),
            },
            device_y=self.bob,
            model_fuse=create_fuse_model_link_prediction(learning_rate),
        )

        history = sl_model.fit(
            x=[nodes, edges],
            y=None,
            epochs=num_epochs,
            dataset_builder=create_dataset_builder(
                stage="train",
                batch_size=batch_size,
                shuffle=True,
            ),
        )
        print(history)
