from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import compose_sf_cluster_config
from secretflow.kuscia.task_config import KusicaTaskConfig
from secretflow.protos.component.cluster_pb2 import SFClusterDesc, StorageConfig
from kuscia.proto.api.v1alpha1.kusciatask.kuscia_task_pb2 import (
    AllocatedPorts,
    ClusterDefine,
    Party,
    Port,
    Service,
)


def test_load_configs():
    kusica_request_json = {
        "task_id": "secretflow-task-20230511100309-single-psi",
        "task_cluster_def": '{"parties":[{"name":"alice","services":[{"port_name":"fed","endpoints":["secretflow-task-20230511100309-single-psi-0-fed.alice.svc"]},{"port_name":"global","endpoints":["secretflow-task-20230511100309-single-psi-0-global.alice.svc:8081"]},{"port_name":"spu","endpoints":["secretflow-task-20230511100309-single-psi-0-spu.alice.svc"]}]},{"name":"bob","services":[{"port_name":"spu","endpoints":["secretflow-task-20230511100309-single-psi-0-spu.bob.svc"]},{"port_name":"fed","endpoints":["secretflow-task-20230511100309-single-psi-0-fed.bob.svc"]},{"port_name":"global","endpoints":["secretflow-task-20230511100309-single-psi-0-global.bob.svc:8081"]}]}]}',
        "allocated_ports": '{"ports":[{"name":"spu","port":54509,"scope":"Cluster","protocol":"GRPC"},{"name":"fed","port":8080,"scope":"Cluster","protocol":"GRPC"},{"name":"global","port":8081,"scope":"Domain","protocol":"GRPC"}]}',
    }

    kuscia_config = KusicaTaskConfig.from_json(kusica_request_json)

    assert kuscia_config.task_id == "secretflow-task-20230511100309-single-psi"
    assert len(kuscia_config.task_cluster_def.parties) == 2
    assert len(kuscia_config.task_allocated_ports.ports) == 3

    ray_config = RayConfig.from_kuscia_task_config(kuscia_config)
    assert (
        ray_config.ray_node_ip_address
        == "secretflow-task-20230511100309-single-psi-0-global.alice.svc"
    )
    assert ray_config.ray_gcs_port == 8081


def test_compose_sf_cluster_config():
    sf_cluster_desc = SFClusterDesc(
        parties=["alice", "bob", "carol"],
        devices=[
            SFClusterDesc.DeviceDesc(name="spu", type="SPU", parties=["alice", "bob"])
        ],
    )

    kusica_task_cluster_def = ClusterDefine(
        parties=[
            Party(
                name="alice",
            ),
            Party(
                name="bob",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.4"]),
                    Service(port_name="spu", endpoints=["1.2.3.5"]),
                ],
            ),
            Party(
                name="carol",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.6:2345"]),
                ],
            ),
        ],
        self_party_idx=0,
    )

    kusica_task_allocated_ports = AllocatedPorts(
        ports=[Port(name="fed", port=1234), Port(name="spu", port=1235)]
    )

    ray_config = RayConfig(ray_node_ip_address="0.0.0.0", ray_gcs_port=1236)
    sf_storage_config = {
        "alice": StorageConfig(
            type="local_fs", local_fs=StorageConfig.LocalFSConfig(wd="/tmp/alice")
        )
    }

    sf_cluster_config = compose_sf_cluster_config(
        sf_cluster_desc,
        "datamesh.local",
        kusica_task_cluster_def,
        kusica_task_allocated_ports,
        ray_config,
        sf_storage_config,
    )

    assert list(sf_cluster_config.public_config.rayfed_config.addresses) == [
        "0.0.0.0:1234",
        "1.2.3.4:80",
        "1.2.3.6:2345",
    ]

    assert list(sf_cluster_config.public_config.spu_configs[0].addresses) == [
        "0.0.0.0:1235",
        "http://1.2.3.5:80",
    ]

    assert sf_cluster_config.private_config.self_party == "alice"
    assert sf_cluster_config.private_config.ray_head_addr == "0.0.0.0:1236"
    assert sf_cluster_config.private_config.storage_config.local_fs.wd == "/tmp/alice"
