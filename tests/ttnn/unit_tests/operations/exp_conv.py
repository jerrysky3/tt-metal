import torch
import ttnn


def main(device):
    # torch_input = torch.rand((1, 1, 28, 28), dtype=torch.bfloat16)
    # torch_weight = torch.rand((16, 1, 3, 3), dtype=torch.bfloat16)

    torch_input = torch.rand((1, 16, 512, 512), dtype=torch.bfloat16)
    torch_weight = torch.rand((32, 16, 6, 6), dtype=torch.bfloat16)

    print(torch.conv2d(torch_input, torch_weight, stride=2, padding=2, dilation=1, groups=1))

    torch_input = torch.permute(torch_input, (0, 2, 3, 1))

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    weight_tensor = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    # input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
    # input_tensor = ttnn.reshape(input_tensor, (1, 1, 28 * 28, 16))
    output_tensor = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight_tensor,
        batch_size=1,
        in_channels=4,
        out_channels=32,
        input_height=512,
        input_width=512,
        kernel_size=(6, 6),
        stride=(2, 2),
        padding=(2, 2),
        dilation=(1, 1),
        device=device,
    )
    print(output_tensor)

    # Output shape: (1, 1, 196[244], C), sharded
    # output_tensor = ttnn.sharded_to_interleaved(output_tensor)
    # # Output shape: (1, 1, 196[244], C), interleaved
    # # Ideally we want to reshape back directly into (1, 14, 14, 16), but it will crash currently. I suspect it is due to the padding 224
    # # output_tensor = ttnn.reshape(output_tensor, (1, 14, 14, 16))

    # # As a workaround, we can move tensor back to host to remove the padding and the following reshape works
    # output_tensor = ttnn.from_device(output_tensor)
    # output_tensor = ttnn.reshape(output_tensor, (1, 14, 14, 16))
    # output_tensor = ttnn.to_device(output_tensor, device=device)
    # output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
    # print(output_tensor.shape)

    # expected_output = torch.conv2d(torch_tensor, (2, 2), (2, 2), (0, 0), (1, 1))
    # assert torch.allclose(ttnn.to_torch(output_tensor), expected_output)


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=65536 * 4)
    try:
        main(device)
    finally:
        ttnn.close_device(device)
