/*
 * Copyright (c) 2022 NVIDIA Corporation
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
#pragma once

#include "../../stdexec/execution.hpp"
#include <optional>

// TODO: Different namespace? STDEXEC_STREAM_DETAIL_NS only for stream scheduler
// stuff?
namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

    // This is a simple inline scheduler for use on a GPU device. It doesn't do
    // much by itself, but serves as a tag type for customizations.
    struct device_scheduler {
      template <class R_>
      struct __op {
        using R = stdexec::__t<R_>;
        [[no_unique_address]] R rec_;

        friend void tag_invoke(stdexec::start_t, __op& op) noexcept {
          stdexec::set_value((R&&) op.rec_);
        }
      };

      struct __sender {
        using is_sender = void;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        template <class R>
        friend auto tag_invoke(stdexec::connect_t, __sender, R&& rec) //
          noexcept
            -> __op<stdexec::__x<stdexec::__decay_t<R>>> {
          return {(R&&) rec};
        }

        struct __env {
          friend device_scheduler
            tag_invoke(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, const __env&) //
            noexcept {
            return {};
          }
        };

        friend __env tag_invoke(stdexec::get_env_t, const __sender&) noexcept {
          return {};
        }
      };

      friend __sender tag_invoke(stdexec::schedule_t, const device_scheduler&) noexcept {
        return {};
      }

      friend stdexec::forward_progress_guarantee
        tag_invoke(stdexec::get_forward_progress_guarantee_t, const device_scheduler&) noexcept {
        return stdexec::forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const device_scheduler&) const noexcept = default;

      template <class Data>
      struct sync_wait_receiver : stdexec::receiver_adaptor<sync_wait_receiver<Data>> {
        Data& data_;

        sync_wait_receiver(Data& data) : data_(data) {}

        template <class... _As>
        void set_value(_As&&... as) noexcept {
          data_.template emplace<1>((_As) as...);
        }

        template <class _E>
        void set_error(_E&&) noexcept {
          // TODO: We can't throw on a GPU. Should sync_wait return a
          // std::expected instead?
          assert(false);
        }

        void set_stopped() noexcept {
          data_.emplace<2>();
        }

        stdexec::empty_env get_env() const {
          return {};
        }
      };

      template <stdexec::sender S>
      friend std::optional<stdexec::value_types_of_t<S, stdexec::empty_env, stdexec::__decayed_tuple, stdexec::__msingle>>
      tag_invoke(stdexec::sync_wait_t, const device_scheduler& self, S&& sndr) {
        using value_t = stdexec::value_types_of_t<S, stdexec::empty_env, stdexec::__decayed_tuple, stdexec::__msingle>;
        using data_t = std::variant<std::monostate, value_t, stdexec::set_stopped_t>;
        data_t __data{};
        using receiver_t = sync_wait_receiver<data_t>;

        stdexec::operation_state auto __op_state = stdexec::connect((S&&) sndr, receiver_t{__data});
        stdexec::start(__op_state);

        // TODO: Can this do anything but inline execution? We expect the result
        // to have been filled in by start.
        // return std::nullopt;
        assert(__data.index() == 1 || __data.index() == 2);

        if (__data.index() == 2)
         return std::nullopt;

        return std::move(std::get<1>(__data));
      }
    };
}
