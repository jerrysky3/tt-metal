// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t output_cb = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_stride = get_compile_time_arg_val(2);

    constexpr uint32_t input_0_num_pages_per_stick = get_compile_time_arg_val(3);
    constexpr uint32_t input_0_num_sticks = get_compile_time_arg_val(4);
    constexpr uint32_t input_0_write_offset = get_compile_time_arg_val(5);
    constexpr uint32_t input_0_read_offset = get_compile_time_arg_val(6);

    constexpr uint32_t input_1_num_pages_per_stick = get_compile_time_arg_val(7);
    constexpr uint32_t input_1_num_sticks = get_compile_time_arg_val(8);
    constexpr uint32_t input_1_write_offset = get_compile_time_arg_val(9);
    constexpr uint32_t input_1_read_offset = get_compile_time_arg_val(10);

    const uint32_t base_l1_write_addr = get_write_ptr(output_cb);

    uint32_t l1_write_addr_0 = base_l1_write_addr + input_0_write_offset;
    const uint32_t l1_read_addr_0 = get_read_ptr(0) + input_0_read_offset;
    const uint64_t noc_addr_0 = get_noc_addr(l1_read_addr_0);
    noc_async_read_one_packet_set_state(noc_addr_0, page_size);

    uint32_t read_offset_0 = l1_read_addr_0;
    for (uint32_t stick_idx = 0; stick_idx < input_0_num_sticks; stick_idx++) {
        for (uint32_t page_idx = 0; page_idx < input_0_num_pages_per_stick; page_idx++) {
            noc_async_read_one_packet_with_state<true>(read_offset_0, l1_write_addr_0 + page_size * page_idx);
            read_offset_0 += page_size;
        }
        l1_write_addr_0 += output_stride;
    }

    uint32_t l1_write_addr_1 = base_l1_write_addr + input_1_write_offset;
    const uint32_t l1_read_addr_1 = get_read_ptr(1) + input_1_read_offset;
    const uint64_t noc_addr_1 = get_noc_addr(l1_read_addr_1);
    noc_async_read_one_packet_set_state(noc_addr_1, page_size);

    uint32_t read_offset_1 = l1_read_addr_1;
    for (uint32_t stick_idx = 0; stick_idx < input_1_num_sticks; stick_idx++) {
        for (uint32_t page_idx = 0; page_idx < input_1_num_pages_per_stick; page_idx++) {
            noc_async_read_one_packet_with_state<true>(read_offset_1, l1_write_addr_1 + page_size * page_idx);
            read_offset_1 += page_size;
        }
        l1_write_addr_1 += output_stride;
    }

    noc_async_read_barrier();
}
