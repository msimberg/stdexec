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
#include <exec/static_thread_pool.hpp>
#include <iostream>
#include <stdexec/execution.hpp>

TEST_CASE("bulk_nested compiles", "[adaptors][bulk_nested]") {
  {
    stdexec::sender auto snd = exec::bulk_nested(
        stdexec::just(42),
        10,
        // std::array{10},
        [](// stdexec::scheduler auto sch,
           int i, int &x) {
          std::cerr << "hello from outer index " << i
                    << " with inline scheduler\n";
          // stdexec::sync_wait(
          //     stdexec::schedule(sch) |
          //     exec::bulk_nested(
          //         std::array{3}, [](stdexec::scheduler auto, int j) {
          //           std::cerr << "hello from inner index " << j
          //                     << " with subscheduler of inline scheduler\n";
          //         }));
        });
    stdexec::sync_wait(std::move(snd));
  }

  // {
  //   stdexec::sender auto snd = exec::bulk_nested(
  //       // Levels 2 onwards are completely ignored in the default case. They
  //       // don't actually require any setup in the default case. Should they be
  //       // allowed?
  //       stdexec::just(42), std::array{10, 17, 7},
  //       [](stdexec::scheduler auto sch, int i, int &x) {
  //         std::cerr << "hello from outer index " << i
  //                   << " with inline scheduler\n";
  //         stdexec::sync_wait(
  //             stdexec::schedule(sch) |
  //             // Can specify any hierarchy here again. The second level from
  //             // above is completely decoupled (ignored) from the level
  //             // specified above. That's ok!?
  //             exec::bulk_nested(
  //                 std::array{3}, [](stdexec::scheduler auto, int j) {
  //                   std::cerr << "hello from inner index " << j
  //                             << " with subscheduler of inline scheduler\n";
  //                 }));
  //       });
  //   stdexec::sync_wait(std::move(snd));
  // }
}

// TEST_CASE("bulk_nested compiles with thread pool scheduler",
//           "[adaptors][bulk_nested]") {
//   exec::static_thread_pool pool_{2};

//   stdexec::sender auto snd =
//       stdexec::transfer_just(pool_.get_scheduler(), 42) |
//       // Levels 2- are completely ignored. They don't actually require any
//       // setup in the default case. Should they be allowed?
//       exec::bulk_nested(std::array{10, 17, 7}, [](stdexec::scheduler auto sch,
//                                                   int i, int &x) {
//         std::cerr << "hello from outer index " << i
//                   << " with static thread pool scheduler\n";
//         stdexec::sync_wait(
//             stdexec::schedule(sch) |
//             exec::bulk_nested(std::array{3}, [](stdexec::scheduler auto,
//                                                 int j) {
//               std::cerr
//                   << "hello from inner index " << j
//                   << " with subscheduler of static thread pool scheduler\n ";
//             }));
//       });
//   stdexec::sync_wait(std::move(snd));
// }
