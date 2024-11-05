import torch
import ttnn


def main(device):
    torch_tensor = torch.rand((1, 16, 28, 28), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.reshape(input_tensor, (1, 1, 28 * 28, 16))
    output_tensor = ttnn.max_pool2d(input_tensor, 1, 28, 28, 16, (2, 2), (2, 2), (0, 0), (1, 1))
    # Output shape: (1, 1, 196[244], C), sharded
    output_tensor = ttnn.sharded_to_interleaved(output_tensor)
    # Output shape: (1, 1, 196[244], C), interleaved
    # Ideally we want to reshape back directly into (1, 14, 14, 16), but it will crash currently. I suspect it is due to the padding 224
    # output_tensor = ttnn.reshape(output_tensor, (1, 14, 14, 16))

    # As a workaround, we can move tensor back to host to remove the padding and the following reshape works
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.reshape(output_tensor, (1, 14, 14, 16))
    output_tensor = ttnn.to_device(output_tensor, device=device)
    output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
    print(output_tensor.shape)

    expected_output = torch.max_pool2d(torch_tensor, (2, 2), (2, 2), (0, 0), (1, 1))
    assert torch.allclose(ttnn.to_torch(output_tensor), expected_output)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        main(device)
    finally:
        ttnn.close_device(device)
