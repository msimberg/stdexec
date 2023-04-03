/*
 * Copyright (c) 2023 ETH Zurich
 * Copyright (c) 2022 Lucian Radu Teodorescu
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// #include <exception>

#include <catch2/catch.hpp>
#include <exec/bulk_nested.hpp>
#include <iostream>
#include "nvexec/stream_context.cuh"
#include <stdexec/execution.hpp>

TEST_CASE("bulk_nested compiles with stream context",
          "[adaptors][bulk_nested]") {
  using nvexec::is_on_gpu;

  nvexec::stream_context stream_ctx{};

  auto bulk_nested_fn = [](stdexec::scheduler auto sch,
                           int i, int &x) {
      printf("hello from outer index %d with stream scheduler (gpu? %d)\n", i, is_on_gpu());
      // TODO: This needs further customizations (especially for sync_wait).
      // stdexec::sync_wait(
      //     stdexec::schedule(sch) |
      //     exec::bulk_nested(std::array{3}, [](stdexec::scheduler auto,
      //                                         int j) {
      //         printf("hello from inner index %d with stream scheduler\n", j);
      //     }));
  };

  stdexec::sender auto snd =
      stdexec::transfer_just(stream_ctx.get_scheduler(), 42) |
      exec::bulk_nested(std::array{10, 17, 7}, bulk_nested_fn);
  stdexec::sync_wait(std::move(snd));
}
