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

import pandas as pd

from secretflow.component.core import (
    SS_GLM_MODEL_MAX,
    Component,
    CompVDataFrame,
    Context,
    DistDataType,
    Field,
    Input,
    Interval,
    Model,
    Output,
    Reporter,
    ServingBuilder,
    SPURuntimeConfig,
    VTable,
    VTableFieldKind,
    register,
    uuid4,
)
from secretflow.device import SPUObject, reveal
from secretflow.ml.linear import SSGLM
from secretflow.ml.linear.ss_glm.model import STOPPING_METRICS

from .ss_glm import SSGLMExportMixin


@register(domain="ml.train", version="1.0.0", name="ss_glm_train")
class SSGLMTrain(SSGLMExportMixin, Component):
    '''
    generalized linear model (GLM) is a flexible generalization of ordinary linear regression.
    The GLM generalizes linear regression by allowing the linear model to be related to the response
    variable via a link function and by allowing the magnitude of the variance of each measurement to
    be a function of its predicted value.
    '''

    epochs: int = Field.attr(
        desc="The number of complete pass through the training data.",
        default=10,
        bound_limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    learning_rate: float = Field.attr(
        desc="The step size at each iteration in one iteration.",
        default=0.1,
        bound_limit=Interval.open(0, None),
        is_checkpoint=True,
    )
    batch_size: int = Field.attr(
        desc="The number of training examples utilized in one iteration.",
        default=1024,
        bound_limit=Interval.open(0, None),
        is_checkpoint=True,
    )
    link_type: str = Field.attr(
        desc="link function type",
        choices=["Logit", "Log", "Reciprocal", "Identity"],
        is_checkpoint=True,
    )
    label_dist_type: str = Field.attr(
        desc="label distribution type",
        choices=["Bernoulli", "Poisson", "Gamma", "Tweedie"],
        is_checkpoint=True,
    )
    tweedie_power: float = Field.attr(
        desc="Tweedie distribution power parameter",
        default=1,
        bound_limit=Interval.closed(0, 2),
        is_checkpoint=True,
    )
    dist_scale: float = Field.attr(
        desc="A guess value for distribution's scale",
        default=1,
        bound_limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    iter_start_irls: int = Field.attr(
        desc="run a few rounds of IRLS training as the initialization of w, 0 disable",
        default=0,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    decay_epoch: int = Field.attr(
        desc="decay learning interval",
        default=0,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    decay_rate: float = Field.attr(
        desc="decay learning rate",
        default=0.0,
        bound_limit=Interval.closed_open(0, 1),
        is_checkpoint=True,
    )
    optimizer: str = Field.attr(
        desc="which optimizer to use: IRLS(Iteratively Reweighted Least Squares) or SGD(Stochastic Gradient Descent)",
        choices=["SGD", "IRLS"],
        is_checkpoint=True,
    )
    l2_lambda: float = Field.attr(
        desc="L2 regularization term",
        default=0.1,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )
    infeed_batch_size_limit: int = Field.attr(
        desc="""
        size of a single block, default to 8w * 100. increase the size will increase memory cost,
        but may decrease running time. Suggested to be as large as possible. (too large leads to OOM)
        """,
        default=8000000,
        bound_limit=Interval.closed(1000, 8000000),
    )
    fraction_of_validation_set: float = Field.attr(
        desc="fraction of training set to be used as the validation set. ineffective for 'weight' stopping_metric",
        default=0.2,
        bound_limit=Interval.open(0, 1),
        is_checkpoint=True,
    )
    random_state: int = Field.attr(
        desc="random state for validation split",
        default=1212,
        bound_limit=Interval.closed(0, None),
        is_checkpoint=True,
    )

    stopping_metric: str = Field.attr(
        desc=f"""
        use what metric as the condition for early stop?  Must be one of {STOPPING_METRICS}.
        only logit link supports AUC metric (note that AUC is very, very expansive in MPC)
        """,
        default='deviance',
        choices=STOPPING_METRICS,
        is_checkpoint=True,
    )
    stopping_rounds: int = Field.attr(
        desc="""
        If the model is not improving for stopping_rounds, the training process will be stopped,
        for 'weight' stopping metric, stopping_rounds is fixed to be 1
        """,
        default=0,
        bound_limit=Interval.closed(0, 100),
        is_checkpoint=True,
    )
    stopping_tolerance: float = Field.attr(
        desc="""
        the model is considered as not improving, if the metric is not improved by tolerance over best metric in history.
        If metric is 'weight' and tolerance == 0, then early stop is disabled.
        """,
        default=0.001,
        bound_limit=Interval.closed_open(0, 1),
        is_checkpoint=True,
    )
    report_metric: bool = Field.attr(
        desc="""
        Whether to report the value of stopping metric.
        Only effective if early stop is enabled.
        If this option is set to true, metric will be revealed and logged.""",
        default=False,
    )
    use_high_precision_exp: bool = Field.attr(
        desc="""
        If you do not know the details of this parameter, please do not modify this parameter!
        If this option is true, glm training and prediction will use a high-precision exp approx,
        but there will be a large performance drop. Otherwise, use high performance exp approx,
        There will be no significant difference in model performance.
        However, prediction bias may occur if the model is exported to an external system for use.
        """,
        default=False,
    )
    exp_iters: int = Field.attr(
        desc="""If you do not know the details of this parameter, please do not modify this parameter!
    Specify the number of iterations of exp taylor approx,
    Only takes effect when use_high_precision_exp is false.
    Increasing this value will improve the accuracy of exp approx,
    but will quickly degrade performance.""",
        default=8,
        bound_limit=Interval.closed(4, 32),
    )
    report_weights: bool = Field.attr(
        desc="If this option is set to true, model will be revealed and model details are visible to all parties",
        default=False,
    )
    feature_selects: list[str] = Field.table_column_attr(
        "input_ds",
        desc="which features should be used for training.",
        limit=Interval.closed(1, None),
        is_checkpoint=True,
    )
    offset: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Specify a column to use as the offset",
        limit=Interval.closed(0, 1),
        is_checkpoint=True,
    )
    weight: list[str] = Field.table_column_attr(
        "input_ds",
        desc="Specify a column to use for the observation weights",
        limit=Interval.closed(0, 1),
        is_checkpoint=True,
    )
    label: str = Field.table_column_attr(
        "input_ds",
        desc="Label of train dataset.",
        is_checkpoint=True,
    )
    input_ds: Input = Field.input(  # type: ignore
        desc="Input vertical table.",
        types=[DistDataType.VERTICAL_TABLE],
        is_checkpoint=True,
    )
    output_model: Output = Field.output(
        desc="Output model.",
        types=[DistDataType.SS_GLM_MODEL],
    )
    report: Output = Field.output(
        desc="If report_weights is true, report model details",
        types=[DistDataType.REPORT],
    )

    def evaluate(self, ctx: Context):
        spu_rt_config = SPURuntimeConfig(
            field="FM128",
            fxp_fraction_bits=40,
            fxp_exp_mode=1 if self.use_high_precision_exp else 2,
            fxp_exp_iters=self.exp_iters,
        )
        spu = ctx.make_spu(config=spu_rt_config)

        checkpoint = None
        cp_dd = ctx.load_checkpoint(
            DistDataType.SS_GLM_CHECKPOINT, SS_GLM_MODEL_MAX, spu=spu
        )
        if cp_dd is not None:
            spu_objs = cp_dd.objs
            train_state = json.loads(cp_dd.public_info)
            checkpoint = (train_state, spu_objs)

        def save_checkpoint_cb(epoch, cp: tuple[dict, list[SPUObject]]):
            cp_uri = f"{self.output_model.uri}_checkpoint_{epoch}"
            train_state, spu_objs = cp
            model = Model(
                "ss_glm",
                DistDataType.SS_GLM_CHECKPOINT,
                SS_GLM_MODEL_MAX,
                objs=spu_objs,
                public_info=json.dumps(train_state),
                system_info=self.input_ds.system_info,
            )
            ctx.dump_checkpoint(epoch, model, cp_uri)

        assert (
            self.label not in self.feature_selects
        ), f"col {self.label} used in both label and features"

        tbl = VTable.from_distdata(self.input_ds)

        tbl_y = tbl.select([self.label])
        tbl_y.check_kinds(VTableFieldKind.FEATURE_LABEL)
        y = ctx.load_table(tbl_y).to_pandas()

        tbl_x = tbl.select(self.feature_selects)
        tbl_x.check_kinds(VTableFieldKind.FEATURE_LABEL)
        x = ctx.load_table(tbl_x).to_pandas()

        if self.offset:
            assert (
                self.offset[0] not in self.feature_selects
            ), f"col {self.offset[0]} used in both offset and features"
            offset_tbl = tbl.select(self.offset)
            offset_tbl.check_kinds(VTableFieldKind.FEATURE_LABEL)
            offset = ctx.load_table(offset_tbl).to_pandas()
        else:
            offset = None

        if self.weight:
            assert (
                self.weight[0] not in self.feature_selects
            ), f"col {self.weight[0]} used in both weight and features"
            weight_tbl = tbl.select(self.weight)
            weight_tbl.check_kinds(VTableFieldKind.FEATURE_LABEL)
            weight = ctx.load_table(weight_tbl).to_pandas()
        else:
            weight = None

        l2_lambda = self.l2_lambda if self.l2_lambda > 0 else None

        glm = SSGLM(spu)
        with ctx.trace_running():
            if self.optimizer == "SGD":
                if self.decay_epoch == 0 or self.decay_rate == 0:
                    self.decay_rate = None
                    self.decay_epoch = None

                glm.fit_sgd(
                    x=x,
                    y=y,
                    offset=offset,
                    weight=weight,
                    epochs=self.epochs,
                    link=self.link_type,
                    dist=self.label_dist_type,
                    tweedie_power=self.tweedie_power,
                    scale=self.dist_scale,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    iter_start_irls=self.iter_start_irls,
                    decay_epoch=self.decay_epoch,
                    decay_rate=self.decay_rate,
                    l2_lambda=l2_lambda,
                    infeed_batch_size_limit=self.infeed_batch_size_limit,
                    fraction_of_validation_set=self.fraction_of_validation_set,
                    random_state=self.random_state,
                    stopping_metric=self.stopping_metric,
                    stopping_rounds=self.stopping_rounds,
                    stopping_tolerance=self.stopping_tolerance,
                    report_metric=self.report_metric,
                    epoch_callback=(
                        save_checkpoint_cb if ctx.enable_checkpoint else None
                    ),
                    recovery_checkpoint=checkpoint,
                )
            elif self.optimizer == "IRLS":
                glm.fit_irls(
                    x=x,
                    y=y,
                    offset=offset,
                    weight=weight,
                    epochs=self.epochs,
                    link=self.link_type,
                    dist=self.label_dist_type,
                    tweedie_power=self.tweedie_power,
                    scale=self.dist_scale,
                    l2_lambda=l2_lambda,
                    infeed_batch_size_limit=self.infeed_batch_size_limit,
                    fraction_of_validation_set=self.fraction_of_validation_set,
                    random_state=self.random_state,
                    stopping_metric=self.stopping_metric,
                    stopping_rounds=self.stopping_rounds,
                    stopping_tolerance=self.stopping_tolerance,
                    report_metric=self.report_metric,
                    epoch_callback=(
                        save_checkpoint_cb if ctx.enable_checkpoint else None
                    ),
                    recovery_checkpoint=checkpoint,
                )
            else:
                raise ValueError(f"Unknown optimizer {self.optimizer}")

        feature_names = x.columns
        party_features_length = {
            device.party: len(columns)
            for device, columns in x.partition_columns.items()
        }

        model_meta = {
            "link": glm.link.link_type().value,
            "dist": glm.dist.dist_type().value,
            "tweedie_power": self.tweedie_power,
            "y_scale": glm.y_scale,
            "offset_col": self.offset if self.offset else [],
            "label_col": [self.label],
            "sample_weight_col": self.weight if self.weight else [],
            "feature_names": feature_names,
            "party_features_length": party_features_length,
            "model_hash": uuid4(tbl.party(0).party),
            "fxp_exp_mode": spu_rt_config.fxp_exp_mode,
            "fxp_exp_iters": spu_rt_config.fxp_exp_iters,
        }

        model_dd = Model(
            "ss_glm",
            DistDataType.SS_GLM_MODEL,
            SS_GLM_MODEL_MAX,
            objs=[glm.spu_w],
            public_info=json.dumps(model_meta),
            system_info=self.input_ds.system_info,
        )
        ctx.dump_to(model_dd, self.output_model)
        self.dump_report(glm, x)

    def dump_report(self, glm: SSGLM, x: CompVDataFrame):
        r = Reporter(
            name="weights and metrics", desc="model weights report and metrics report"
        )
        if self.report_weights:
            weights = list(map(float, list(reveal(glm.spu_w))))
            named_weight = {}
            for p in x.partitions.values():
                features = p.columns
                party_weight = weights[: len(features)]
                named_weight.update({f: w for f, w in zip(features, party_weight)})
                weights = weights[len(features) :]
            assert len(weights) == 1

            w_desc = {"_intercept_": weights[-1], "_y_scale_": glm.y_scale}
            for f, w in named_weight.items():
                w_desc[f] = w
            r.add_tab(w_desc, name="weights", desc="model weights")

        effective_train = (
            self.stopping_metric == 'weight' and self.stopping_tolerance > 0
        ) or (self.stopping_metric != 'weight' and self.stopping_rounds > 0)

        if self.report_metric and effective_train:
            metric_logs = glm.train_metric_history
            assert isinstance(metric_logs, list)
            assert len(metric_logs) >= 1, "must train the model for at least 1 round"
            df = pd.DataFrame(metric_logs, dtype=str)
            r.add_tab(
                df,
                name="metrics log",
                desc="metrics for training and validation set at each epoch (indexed from 1)",
            )

        r.dump_to(self.report, self.input_ds.system_info)

    def export(self, ctx: Context, builder: ServingBuilder, he_mode: bool) -> None:
        return self.do_export(
            ctx, builder, self.input_ds, self.output_model.data, he_mode
        )
