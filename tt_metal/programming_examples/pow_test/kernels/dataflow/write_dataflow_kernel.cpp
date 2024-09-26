// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    constexpr uint32_t cb_id_out0 = 16;
    constexpr uint32_t num_of_tiles = 1;
    const uint32_t block_bytes = get_tile_size(cb_id_out0) * num_of_tiles;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t num_of_blocks = get_arg_val<uint32_t>(3);

    for (uint32_t idx = 0; idx < num_of_blocks; ++idx) {
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr);
        cb_wait_front(cb_id_out0, num_of_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write(l1_read_addr, dst_noc_addr, block_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, num_of_tiles);
        dst_addr += block_bytes;
    }

    DPRINT_DATA1(DPRINT << "NOC 1 done" << ENDL());
}
