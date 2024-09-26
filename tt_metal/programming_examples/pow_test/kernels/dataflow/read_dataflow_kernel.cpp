// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

void kernel_main() {
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t num_of_tiles = 1;
    const uint32_t block_bytes = get_tile_size(cb_id_in0) * num_of_tiles;

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t src_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t num_of_blocks = get_arg_val<uint32_t>(3);

    for (uint32_t idx = 0; idx < num_of_blocks; ++idx) {
        uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
        cb_reserve_back(cb_id_in0, num_of_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(src_noc_addr, l1_write_addr, block_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, num_of_tiles);
        src_addr += block_bytes;
    }

    DPRINT_DATA0(DPRINT << "NOC 0 done " << ENDL());
}
