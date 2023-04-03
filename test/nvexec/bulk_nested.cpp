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

  // TODO: Is all of the below correct?

  // TODO: The device-side nested bulks currently manually compute the correct
  // indices. Replace by algorithm customizations.
  auto bulk_nested_fn = [](stdexec::scheduler auto sch,
                           int i, int &x) {
      printf("hello from outer index %d with stream scheduler (gpu? %d)\n", i, is_on_gpu());
      stdexec::sync_wait(
        stdexec::schedule(sch) |
        // TODO: Should this only be printed once per team? Separate algorithm?
        // stdexec::once? stdexec::single? Currently it prints team_size times.
        stdexec::then([] {
          if (threadIdx.x == 0) {
            printf("hello from inner stream scheduler then\n");
          }
        }) |
        // This should use the parallelism in the second level to print hello 9
        // times. If this level is using 4 CUDA threads, those 4 CUDA threads
        // will each loop at most 3 times to cover the iteration space of 9.
        stdexec::bulk(9, [=](int j) {
          for (int j_actual = threadIdx.x; j_actual < 9; j_actual += blockDim.x) {
            printf("hello from outer index %d, inner index %d with stream scheduler\n", i, j_actual);
          }
        }) |
        // This should use the parallelism in the second level to print hello 9
        // times. If this level is using 4 CUDA threads, those 4 CUDA threads
        // will each loop at most 3 times to cover the iteration space of 9.
        exec::bulk_nested(std::array{9}, [=](stdexec::scheduler auto, int j) {
          for (int j_actual = threadIdx.x; j_actual < 9; j_actual += blockDim.x) {
            printf("hello from outer index %d, inner index %d with stream scheduler\n", i, j_actual);
            // The last level (7) was ignored. Simply run with inline_scheduler.
            // This will run as many times as the above print times 2.
            stdexec::sync_wait(
              stdexec::schedule(sch) |
              stdexec::bulk(2, [=](int k) {
                printf("hello from outer index %d, inner index %d, innermost index %d with stream scheduler\n", i, j_actual, k);
              }));
          }
        }));
  };

  stdexec::sender auto snd =
      stdexec::transfer_just(stream_ctx.get_scheduler(), 42) |
      exec::bulk_nested(std::array{3, 4, 7}, bulk_nested_fn);
  stdexec::sync_wait(std::move(snd));
}
