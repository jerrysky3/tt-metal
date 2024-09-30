import torch
import ttnn


def main(device):
    input_data = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    x = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    input_shard_config = ttnn.create_sharded_memory_config(
        shape=x.shape,
        core_grid=ttnn.CoreGrid(y=2, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    x_t = ttnn.to_memory_config(x, memory_config=input_shard_config, dtype=ttnn.bfloat16)

    output_shard_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, 64, 32],
        core_grid=ttnn.CoreGrid(y=2, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    y_t = ttnn.experimental.concat([x_t, x_t], dim=2, memory_config=output_shard_config)
    output_data = ttnn.to_torch(y_t)

    print(input_data, input_data.shape)
    print(output_data, output_data.shape)
    print(torch.allclose(torch.concat([input_data, input_data], dim=2), output_data))


if __name__ == "__main__":
    device_id = 1
    device = ttnn.open_device(device_id=device_id)
    try:
        main(device)
    finally:
        ttnn.close_device(device)
