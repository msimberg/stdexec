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
#include <stdexec/execution.hpp>

TEST_CASE("bulk_nested compiles", "[adaptors][bulk_nested]") {
  stdexec::sender auto snd = exec::bulk_nested(
      stdexec::just(42), 10, [](stdexec::scheduler auto sch, int i, int &x) {
        std::cerr << "hello from outer index " << i << " with x = " << x
                  << '\n';
        stdexec::sync_wait(
            stdexec::schedule(sch) |
            exec::bulk_nested(3, [](stdexec::scheduler auto, int j) {
              std::cerr << "hello from inner index " << j << '\n';
            }));
      });
  stdexec::sync_wait(std::move(snd));
}
