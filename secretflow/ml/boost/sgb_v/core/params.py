# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum, unique

from secretflow.device import HEU, PYU, PYUObject


@unique
class RegType(Enum):
    Linear = 'linear'
    Logistic = 'logistic'


@dataclass
class LabelHolderInfo:
    label_holder_pyu: PYU
    heu: HEU
    y: PYUObject
    seed: int
    reg_lambda: float
    learning_rate: float
    base_score: float
    sample_num: int
    subsample_rate: float
    obj_type: RegType


@dataclass
class SGBTrainParams:
    num_boost_round: int
    max_depth: int
    learning_rate: float
    objective: RegType
    reg_lambda: float
    subsample: float
    colsample_by_tree: float
    base_score: float
    sketch_eps: float
    seed: int
    fixed_point_parameter: int
