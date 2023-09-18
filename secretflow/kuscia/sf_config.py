# Copyright 2023 Ant Group Co., Ltd.
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


from kuscia.proto.api.v1alpha1.kusciatask.kuscia_task_pb2 import (
    AllocatedPorts,
    ClusterDefine,
)

from secretflow.kuscia.datamesh import (
    create_domain_data_source_service_stub,
    get_domain_data_source,
)
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.task_config import KusicaTaskConfig
from secretflow.protos.component.cluster_pb2 import (
    SFClusterConfig,
    SFClusterDesc,
    StorageConfig,
)


def compose_sf_cluster_config(
    sf_cluster_desc: SFClusterDesc,
    datamesh_addr: str,
    kusica_task_cluster_def: ClusterDefine,
    kusica_task_allocated_ports: AllocatedPorts,
    ray_config: RayConfig,
    sf_storage_config: StorageConfig,
    datasource_id: str = None,
) -> SFClusterConfig:
    party_id = kusica_task_cluster_def.self_party_idx
    party_name = kusica_task_cluster_def.parties[party_id].name

    spu_address = {}
    fed_address = {}

    for port in kusica_task_allocated_ports.ports:
        if port.name == "spu":
            spu_address["spu"] = {party_name: f"0.0.0.0:{port.port}"}

        elif port.name == "fed":
            fed_address = {party_name: f"0.0.0.0:{port.port}"}

    for party in kusica_task_cluster_def.parties:
        if party.name != party_name:
            for service in party.services:
                if service.port_name == "fed":
                    if len(service.endpoints[0].split(":")) < 2:
                        service.endpoints[0] += ":80"
                    fed_address[party.name] = service.endpoints[0]
                elif service.port_name == "spu":
                    if "spu" not in spu_address:
                        spu_address["spu"] = {}

                    if len(service.endpoints[0].split(":")) < 2:
                        service.endpoints[0] += ":80"
                    # add "http://" to force brpc to set the correct Host
                    spu_address["spu"][party.name] = f"http://{service.endpoints[0]}"

    if (
        sf_storage_config is not None
        and party_name not in sf_storage_config
        and datasource_id is None
    ):
        raise RuntimeError(
            f"storage config of party [{party_name}] is missing. It must be provided with sf_storage_config explicitly or be inferred from sf_input_ids with DataMesh services."
        )

    if sf_storage_config is not None and party_name in sf_storage_config:
        storage_config = sf_storage_config[party_name]
    else:
        # try to get storage config with sf_datasource_config
        stub = create_domain_data_source_service_stub(datamesh_addr)
        domain_data_source = get_domain_data_source(stub, datasource_id)

        storage_config = StorageConfig(
            type="local_fs",
            local_fs=StorageConfig.LocalFSConfig(
                wd=domain_data_source.info.localfs.path
            ),
        )

    res = SFClusterConfig(
        desc=sf_cluster_desc,
        private_config=SFClusterConfig.PrivateConfig(
            self_party=party_name,
            ray_head_addr=f"{ray_config.ray_node_ip_address}:{ray_config.ray_gcs_port}",
            storage_config=storage_config,
        ),
    )

    public_ray_fed_config = SFClusterConfig.RayFedConfig()
    if set(list(sf_cluster_desc.parties)) != set(list(fed_address.keys())):
        raise RuntimeError(
            "parties in kusica_task doesn't match those in sf_cluster_desc"
        )
    for p in sf_cluster_desc.parties:
        public_ray_fed_config.parties.append(p)
        public_ray_fed_config.addresses.append(fed_address[p])

    res.public_config.ray_fed_config.CopyFrom(public_ray_fed_config)

    for device in sf_cluster_desc.devices:
        if device.type.lower() == "spu":
            if device.name not in spu_address:
                raise RuntimeError(
                    f"addresses of SPU [{device.name}] are not set in kusica_task."
                )

            spu_addr = spu_address[device.name]

            if set(list(device.parties)) != set(list(spu_addr.keys())):
                raise RuntimeError(
                    f"parties of SPU [{device.name}] in kusica_task doesn't match those in sf_cluster_desc"
                )

            pulic_spu_config = SFClusterConfig.SPUConfig(name=device.name)
            for p in device.parties:
                pulic_spu_config.parties.append(p)
                pulic_spu_config.addresses.append(spu_addr[p])

            res.public_config.spu_configs.append(pulic_spu_config)

    return res


def get_sf_cluster_config(
    kuscia_config: KusicaTaskConfig, datamesh_addr: str, datasource_id: str = None
) -> SFClusterConfig:
    return compose_sf_cluster_config(
        kuscia_config.sf_cluster_desc,
        datamesh_addr,
        kuscia_config.task_cluster_def,
        kuscia_config.task_allocated_ports,
        RayConfig.from_kuscia_task_config(kuscia_config),
        kuscia_config.sf_storage_config,
        datasource_id,
    )
