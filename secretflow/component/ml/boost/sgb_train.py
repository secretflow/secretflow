# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from secretflow.component.core import (
    Component,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    IServingExporter,
    Model,
    Output,
    register,
    Reporter,
    ServingBuilder,
    SGB_MODEL_MAX,
    VTable,
    VTableFieldKind,
)
from secretflow.data.vertical.dataframe import VDataFrame
from secretflow.device import PYU
from secretflow.ml.boost.core.callback import TrainingCallback
from secretflow.ml.boost.core.metric import ALL_METRICS_NAMES
from secretflow.ml.boost.sgb_v import Sgb, SgbModel
from secretflow.ml.boost.sgb_v.checkpoint import (
    sgb_model_to_snapshot,
    SGBCheckpointData,
)
from secretflow.ml.boost.sgb_v.core.importance import (
    SUPPORTED_IMPORTANCE_DESCRIPTIONS,
    SUPPORTED_IMPORTANCE_TYPE_STATS,
)
from secretflow.ml.boost.sgb_v.core.params import RegType
from secretflow.ml.boost.sgb_v.factory.booster.global_ordermap_booster import (
    build_checkpoint,
    GlobalOrdermapBooster,
)

from .sgb import SGBExportMixin


@register(domain="ml.train", version="1.1.0", name="sgb_train")
class SGBTrain(SGBExportMixin, Component, IServingExporter):
    '''
    Provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using secure boost.

    - SGB is short for SecureBoost. Compared to its safer counterpart SS-XGB, SecureBoost focused on protecting label holder.

    - Check https://arxiv.org/abs/1901.08755.
    '''

    num_boost_round: int = Field.attr(
        desc="Number of boosting iterations.",
        default=10,
        bound_limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    max_depth: int = Field.attr(
        desc="Maximum depth of a tree.",
        default=5,
        bound_limit=Interval.closed(1, 16),
        is_checkpoint=True,
    )
    learning_rate: float = Field.attr(
        desc="Step size shrinkage used in update to prevent overfitting.",
        default=0.1,
        bound_limit=Interval.open_closed(0.0, 1.0),
        is_checkpoint=True,
    )
    objective: str = Field.attr(
        desc="Specify the learning objective.",
        default="logistic",
        choices=[reg_type.value for reg_type in RegType],
        is_checkpoint=True,
    )
    reg_lambda: float = Field.attr(
        desc="L2 regularization term on weights.",
        default=0.1,
        bound_limit=Interval.closed(0, 10000),
        is_checkpoint=True,
    )
    gamma: float = Field.attr(
        desc="Greater than 0 means pre-pruning enabled. If gain of a node is less than this value, it would be pruned.",
        default=1,
        bound_limit=Interval.closed(0, 10000),
        is_checkpoint=True,
    )
    colsample_by_tree: float = Field.attr(
        desc="Subsample ratio of columns when constructing each tree.",
        default=1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    sketch_eps: float = Field.attr(
        desc="This roughly translates into O(1 / sketch_eps) number of bins.",
        default=0.1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    base_score: float = Field.attr(
        desc="The initial prediction score of all instances, global bias.",
        default=0,
        bound_limit=Interval.closed(-10, 10),
        is_checkpoint=True,
    )
    seed: int = Field.attr(
        desc="Pseudorandom number generator seed.",
        default=42,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    fixed_point_parameter: int = Field.attr(
        desc="""Any floating point number encoded by heu,
            will multiply a scale and take the round,
            scale = 2 ** fixed_point_parameter.
            larger value may mean more numerical accuracy,
            but too large will lead to overflow problem.""",
        default=20,
        bound_limit=Interval.closed(1, 100),
        is_checkpoint=True,
    )
    first_tree_with_label_holder_feature: bool = Field.attr(
        desc="Whether to train the first tree with label holder's own features.",
        default=False,
        is_checkpoint=True,
    )
    batch_encoding_enabled: bool = Field.attr(
        desc="If use batch encoding optimization.",
        default=True,
        is_checkpoint=True,
    )
    enable_quantization: bool = Field.attr(
        desc="Whether enable quantization of g and h.",
        default=False,
        is_checkpoint=True,
    )
    quantization_scale: float = Field.attr(
        desc="Scale the sum of g to the specified value.",
        default=10000.0,
        bound_limit=Interval.closed(0, 10000000.0),
        is_checkpoint=True,
    )
    max_leaf: int = Field.attr(
        desc="Maximum leaf of a tree. Only effective if train leaf wise.",
        default=15,
        bound_limit=Interval.closed(1, 2**15),
    )
    rowsample_by_tree: float = Field.attr(
        desc="Row sub sample ratio of the training instances.",
        default=1,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    enable_goss: bool = Field.attr(
        desc="Whether to enable GOSS.",
        default=False,
        is_checkpoint=True,
    )
    top_rate: float = Field.attr(
        desc="GOSS-specific parameter. The fraction of large gradients to sample.",
        default=0.3,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    bottom_rate: float = Field.attr(
        desc="GOSS-specific parameter. The fraction of small gradients to sample.",
        default=0.5,
        bound_limit=Interval.open_closed(0, 1),
        is_checkpoint=True,
    )
    tree_growing_method: str = Field.attr(
        desc="How to grow tree?",
        default="level",
        is_checkpoint=True,
    )
    enable_early_stop: bool = Field.attr(
        desc="Whether to enable early stop during training.",
        default=False,
        is_checkpoint=True,
    )
    enable_monitor: bool = Field.attr(
        desc="Whether to enable monitoring performance during training.",
        default=False,
        is_checkpoint=True,
    )
    eval_metric: str = Field.attr(
        desc=f"Use what metric for monitoring and early stop? Currently support {ALL_METRICS_NAMES}",
        default="roc_auc",
        choices=ALL_METRICS_NAMES,
        is_checkpoint=True,
    )
    validation_fraction: float = Field.attr(
        desc="Early stop specific parameter. Only effective if early stop enabled. The fraction of samples to use as validation set.",
        default=0.1,
        bound_limit=Interval.open(0, 1),
        is_checkpoint=True,
    )
    stopping_rounds: int = Field.attr(
        desc="""Early stop specific parameter. If more than `stopping_rounds` consecutive rounds without improvement, training will stop.
    Only effective if early stop enabled""",
        default=1,
        bound_limit=Interval.closed(1, 1024),
        is_checkpoint=True,
    )
    stopping_tolerance: float = Field.attr(
        desc="Early stop specific parameter. If metric on validation set is no longer improving by at least this amount, then consider not improving.",
        default=0.0,
        bound_limit=Interval.closed(0.0, None),
        is_checkpoint=True,
    )
    tweedie_variance_power: float = Field.attr(
        desc="Parameter that controls the variance of the Tweedie distribution.",
        default=1.5,
        bound_limit=Interval.open(1.0, 2.0),
        is_checkpoint=True,
    )
    save_best_model: bool = Field.attr(
        desc="Whether to save the best model on validation set during training.",
        default=False,
        is_checkpoint=True,
    )
    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be used for training.",
        limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="Label of train dataset.",
        is_checkpoint=True,
    )

    report_importances: bool = Field.attr(
        desc=f"""Whether to report feature importances.
            Currently supported importances are:
            {json.dumps(SUPPORTED_IMPORTANCE_DESCRIPTIONS)}
        """,
        default=False,
        minor_min=1,
    )

    input_ds: Input = Field.input(
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SGB_MODEL],
    )
    report: Output = Field.output(
        desc="If report_importances is true, report feature importances",
        types=[DistDataType.REPORT],
        minor_min=1,
    )

    def __post_init__(self):
        # back support version 1.0
        self.whether_dump_report = True
        if self._minor == 0:
            self.whether_dump_report = False

    def evaluate(self, ctx: Context):
        # assert ctx.heu_config is not None, "need heu config in SFClusterDesc"
        assert (
            self.label not in self.feature_selects
        ), f"expect no intersection between label and features, got {self.label} and {self.feature_selects}"
        tbl = VTable.from_distdata(self.input_ds)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)
        y = ctx.load_table(tbl_y).to_pandas()
        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE_LABEL)
        x = ctx.load_table(tbl_x).to_pandas()

        label_party = next(iter(y.partitions.keys())).party
        heu_evaluators = [p.party for p in x.partitions if p.party != label_party]
        heu = ctx.make_heu(label_party, heu_evaluators)
        pyus = {p: PYU(p) for p in ctx.parties}

        checkpoint_data = None
        cp_dd = ctx.load_checkpoint(
            DistDataType.SGB_CHECKPOINT, SGB_MODEL_MAX, pyus=pyus
        )
        if cp_dd is not None:
            metas = json.loads(cp_dd.public_info)
            checkpoint_data = SGBCheckpointData(cp_dd.objs, metas)

        def epoch_callback(
            model: GlobalOrdermapBooster,
            epoch: int,
            evals_log: TrainingCallback.EvalsLog,
        ):
            ctx.update_progress((epoch + 1) / self.num_boost_round)
            if not ctx.enable_checkpoint:
                return
            cp_uri = f"{self.output_model.uri}_checkpoint_{epoch}"
            cp = build_checkpoint(model, evals_log, x, [self.label])
            cp_dd = Model(
                "sgb",
                DistDataType.SGB_CHECKPOINT,
                SGB_MODEL_MAX,
                objs=cp.model_objs,
                public_info=json.dumps(cp.model_train_state_metas),
                system_info=self.input_ds.system_info,
            )
            ctx.dump_checkpoint(epoch, cp_dd, cp_uri)

        with ctx.tracer.trace_running():
            sgb = Sgb(heu)
            model = sgb.train(
                params={
                    'num_boost_round': self.num_boost_round,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'objective': self.objective,
                    'reg_lambda': self.reg_lambda,
                    'gamma': self.gamma,
                    'rowsample_by_tree': self.rowsample_by_tree,
                    'colsample_by_tree': self.colsample_by_tree,
                    'bottom_rate': self.bottom_rate,
                    'top_rate': self.top_rate,
                    'max_leaf': self.max_leaf,
                    'quantization_scale': self.quantization_scale,
                    'sketch_eps': self.sketch_eps,
                    'base_score': self.base_score,
                    'seed': self.seed,
                    'fixed_point_parameter': self.fixed_point_parameter,
                    'enable_goss': self.enable_goss,
                    'enable_quantization': self.enable_quantization,
                    'batch_encoding_enabled': self.batch_encoding_enabled,
                    'tree_growing_method': self.tree_growing_method,
                    'first_tree_with_label_holder_feature': self.first_tree_with_label_holder_feature,
                    "enable_monitor": self.enable_monitor,
                    "enable_early_stop": self.enable_early_stop,
                    "eval_metric": self.eval_metric,
                    "validation_fraction": self.validation_fraction,
                    "stopping_rounds": self.stopping_rounds,
                    "stopping_tolerance": self.stopping_tolerance,
                    "save_best_model": self.save_best_model,
                    "tweedie_variance_power": self.tweedie_variance_power,
                },
                dtrain=x,
                label=y,
                checkpoint_data=checkpoint_data,
                dump_function=epoch_callback,
            )

        snapshot = sgb_model_to_snapshot(model, x, [self.label])
        model_db = Model(
            "sgb",
            DistDataType.SGB_MODEL,
            SGB_MODEL_MAX,
            objs=snapshot.model_objs,
            public_info=json.dumps(snapshot.model_meta),
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_db, self.output_model)

        if self.whether_dump_report:
            self.dump_report(model, x)

    def dump_report(self, model: SgbModel, x: VDataFrame):
        r = Reporter(
            name="feature importances",
            desc="""feature importance report for all availble importance types
            """,
            system_info=self.input_ds.system_info,
        )

        if self.report_importances:
            for importance_type in SUPPORTED_IMPORTANCE_TYPE_STATS.keys():
                importances = model.feature_importance_flatten(x, importance_type)
                named_importance = {}
                for p in x.partitions.values():
                    features = p.columns
                    party_importances = importances[: len(features)]
                    named_importance.update(
                        {f: w for f, w in zip(features, party_importances)}
                    )
                    importances = importances[len(features) :]
            assert len(importances) == 0
            r.add_tab(
                named_importance,
                name=f"{importance_type}",
                desc=f"""feature importance of type {importance_type}. 
                {SUPPORTED_IMPORTANCE_DESCRIPTIONS[importance_type]}""",
            )

        self.report.data = r.to_distdata()

    def export(
        self, ctx: Context, builder: ServingBuilder, he_mode: bool = False
    ) -> None:
        return self.do_export(
            ctx, builder, self.input_ds, self.output_model.data, he_mode
        )
