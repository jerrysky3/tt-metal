import torch
import ttnn


def gen_data(shape, grid, strategy):
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    input_shard_config = ttnn.create_sharded_memory_config(
        shape=x.shape,
        core_grid=grid,
        strategy=strategy,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    return input_data, ttnn.to_memory_config(x, memory_config=input_shard_config, dtype=ttnn.bfloat16)


def main_width(device):
    grid = ttnn.CoreGrid(y=2, x=4)
    width = 64
    input_data_0, x_0 = gen_data([1, 1, 8, width], grid, ttnn.ShardStrategy.WIDTH)
    input_data_1, x_1 = gen_data([1, 1, 8, width], grid, ttnn.ShardStrategy.WIDTH)
    input_data_2, x_2 = gen_data([1, 1, 23, width], grid, ttnn.ShardStrategy.WIDTH)
    output_shard_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, 8 + 8 + 23, width],
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    y_t = ttnn.experimental.concat([x_0, x_1, x_2], dim=2, memory_config=output_shard_config)
    output_data = ttnn.to_torch(y_t)

    print(input_data_0, input_data_0.shape)
    print(output_data, output_data.shape)
    print(torch.allclose(torch.concat([input_data_0, input_data_1, input_data_2], dim=2), output_data))


def main_width_cache(device):
    grid = ttnn.CoreGrid(y=2, x=2)
    input_data_0, x_0 = gen_data([1, 1, 8, 64], grid, ttnn.ShardStrategy.WIDTH)
    input_data_1, x_1 = gen_data([1, 1, 8, 64], grid, ttnn.ShardStrategy.WIDTH)
    # input_data_2, x_2 = gen_data([1, 1, 23, 32], grid, ttnn.ShardStrategy.WIDTH)
    output_shard_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, 8 + 8, 64],
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    y_0 = ttnn.experimental.concat([x_0, x_1], dim=2, memory_config=output_shard_config)

    input_data_2, x_2 = gen_data([1, 1, 8, 64], grid, ttnn.ShardStrategy.WIDTH)
    y_1 = ttnn.experimental.concat([x_0, x_2], dim=2, memory_config=output_shard_config)

    print(torch.allclose(torch.concat([input_data_0, input_data_1], dim=2), ttnn.to_torch(y_0)))
    print(torch.allclose(torch.concat([input_data_0, input_data_2], dim=2), ttnn.to_torch(y_1)))


def main_height(device):
    grid = ttnn.CoreGrid(y=1, x=3)
    input_data_0, x_0 = gen_data([1, 1, 9, 16], grid, ttnn.ShardStrategy.HEIGHT)
    input_data_1, x_1 = gen_data([1, 1, 9, 64], grid, ttnn.ShardStrategy.HEIGHT)
    input_data_2, x_2 = gen_data([1, 1, 9, 32], grid, ttnn.ShardStrategy.HEIGHT)
    output_shard_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, 9, 16 + 64 + 32],
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    y_t = ttnn.experimental.concat([x_0, x_1, x_2], dim=3, memory_config=output_shard_config)
    output_data = ttnn.to_torch(y_t)

    print(input_data_0, input_data_0.shape)
    print(output_data, output_data.shape)
    print(torch.allclose(torch.concat([input_data_0, input_data_1, input_data_2], dim=3), output_data))


if __name__ == "__main__":
    device_id = 1
    device = ttnn.open_device(device_id=device_id)
    device.enable_program_cache()
    try:
        main_width(device)
        main_width_cache(device)
        main_height(device)
    finally:
        ttnn.close_device(device)
