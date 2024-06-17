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
#include "../../exec/bulk_nested.hpp"
#include "../detail/device_scheduler.cuh"
#include <type_traits>

#include "common.cuh"

namespace nvexec::STDEXEC_STREAM_DETAIL_NS {

  namespace _bulk {
    template <int BlockThreads, class... As, std::integral Shape, class Fun>
    __launch_bounds__(BlockThreads) __global__ void kernel(Shape shape, Fun fn, As... as) {
      static_assert(trivially_copyable<Shape, Fun, As...>);
      const int tid = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);

      if (tid < static_cast<int>(shape)) {
        ::cuda::std::move(fn)(tid, static_cast<As&&>(as)...);
      }
    }

    template <class ReceiverId, std::integral Shape, class Fun>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = typename operation_state_base_t<ReceiverId>::env_t;

        Shape shape_;
        Fun f_;

        operation_state_base_t<ReceiverId>& op_state_;

       public:
        using __id = receiver_t;

        template <class... As>
          requires __callable<Fun, Shape&, As&...>
        void set_value(As&&... as) noexcept {
          operation_state_base_t<ReceiverId>& op_state = op_state_;

          if (shape_) {
            cudaStream_t stream = op_state.get_stream();
            constexpr int block_threads = 256;
            const int grid_blocks = (static_cast<int>(shape_) + block_threads - 1) / block_threads;
            kernel<block_threads, As&...>
              <<<grid_blocks, block_threads, 0, stream>>>(shape_, std::move(f_), as...);
          }

          if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
            op_state.propagate_completion_signal(stdexec::set_value, static_cast<As&&>(as)...);
          } else {
            op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
          op_state_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept {
          op_state_.propagate_completion_signal(set_stopped_t());
        }

        auto get_env() const noexcept -> Env {
          return op_state_.make_env();
        }

        explicit __t(Shape shape, Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : shape_(shape)
          , f_(static_cast<Fun&&>(fun))
          , op_state_(op_state) {
        }
      };
    };
  } // namespace _bulk

  template <class SenderId, std::integral Shape, class Fun>
  struct bulk_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = bulk_sender_t;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      using _set_error_t = completion_signatures<set_error_t(cudaError_t)>;

      template <class Receiver>
      using receiver_t = stdexec::__t<_bulk::receiver_t<stdexec::__id<Receiver>, Shape, Fun>>;

      template <class... Tys>
      using _set_value_t = completion_signatures<set_value_t(Tys...)>;

      template <class Self, class Env>
      using _completion_signatures_t = //
        __try_make_completion_signatures<
          __copy_cvref_t<Self, Sender>,
          Env,
          _set_error_t,
          __q<_set_value_t>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
      STDEXEC_MEMFN_DECL(
        auto connect)(this Self&& self, Receiver rcvr)
        -> stream_op_state_t<__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<__copy_cvref_t<Self, Sender>>(
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr),
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> {
            return receiver_t<Receiver>(self.shape_, static_cast<Fun&&>(self.fun_), stream_provider);
          });
      }

      template <__decays_to<__t> Self, class Env>
      static auto get_completion_signatures(Self&&, Env&&) -> _completion_signatures_t<Self, Env> {
        return {};
      }

      auto get_env() const noexcept -> env_of_t<const Sender&> {
        return stdexec::get_env(sndr_);
      }
    };
  };

  namespace bulk_nested {

    template <std::integral Shape, class Fun, class... As>
    __global__ void kernel(Shape shape, Fun fn, As... as) {
      fn(device_scheduler{}, blockIdx.x, as...);
    }

    template <class ReceiverId, std::integral Shape, std::size_t N, class Fun>
    struct receiver_t {
      class __t : public stream_receiver_base {
        using Receiver = stdexec::__t<ReceiverId>;
        using Env = typename operation_state_base_t<ReceiverId>::env_t;

        std::array<Shape, N> shape_;
        Fun f_;

        operation_state_base_t<ReceiverId>& op_state_;

       public:
        using __id = receiver_t;

        template <class... As>
        friend void tag_invoke(stdexec::set_value_t, __t&& self, As&&... as) noexcept
          requires stdexec::__callable<Fun, device_scheduler, Shape, As&...>
        {
          operation_state_base_t<ReceiverId>& op_state = self.op_state_;

          if constexpr (N >= 1) {
            if (self.shape_[0]) {
              cudaStream_t stream = op_state.get_stream();
              const int grid_blocks = self.shape_[0];
              const int block_threads = [&]{
                if constexpr (N >= 2) {
                  return self.shape_[1];
                } else {
                  return 1;
                }

                // This is only to silence nvc++
                return 1;
              }();
              kernel<Shape, Fun, As...>
                <<<grid_blocks, block_threads, 0, stream>>>(block_threads, self.f_, (As&&) as...);
            }
          }

          if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
            op_state.propagate_completion_signal(stdexec::set_value, (As&&) as...);
          } else {
            op_state.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.op_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend Env tag_invoke(stdexec::get_env_t, const __t& self) noexcept {
          return self.op_state_.make_env();
        }

        explicit __t(std::array<Shape, N> shape, Fun fun, operation_state_base_t<ReceiverId>& op_state)
          : shape_(shape)
          , f_((Fun&&) fun)
          , op_state_(op_state) {
        }
      };
    };
  }

  template <class SenderId, std::integral Shape, std::size_t N, class Fun>
  struct bulk_nested_sender_t {
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = bulk_nested_sender_t;
      Sender sndr_;
      std::array<Shape, N> shape_;
      Fun fun_;

      using set_error_t = stdexec::completion_signatures< stdexec::set_error_t(cudaError_t)>;

      template <class Receiver>
      using receiver_t = stdexec::__t<bulk_nested::receiver_t<stdexec::__id<Receiver>, Shape, N, Fun>>;

      template <class... Tys>
      using set_value_t = stdexec::completion_signatures< stdexec::set_value_t(Tys...)>;

      template <class Self, class Env>
      using completion_signatures = //
        stdexec::__make_completion_signatures<
          stdexec::__copy_cvref_t<Self, Sender>,
          Env,
          set_error_t,
          stdexec::__q<set_value_t>>;

      template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
        requires stdexec::receiver_of<
          Receiver,
          completion_signatures<Self, stdexec::env_of_t<Receiver>>>
      friend auto tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr)
        -> stream_op_state_t<stdexec::__copy_cvref_t<Self, Sender>, receiver_t<Receiver>, Receiver> {
        return stream_op_state<stdexec::__copy_cvref_t<Self, Sender>>(
          ((Self&&) self).sndr_,
          (Receiver&&) rcvr,
          [&](operation_state_base_t<stdexec::__id<Receiver>>& stream_provider)
            -> receiver_t<Receiver> {
            return receiver_t<Receiver>(self.shape_, self.fun_, stream_provider);
          });
      }

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> stdexec::dependent_completion_signatures<Env>;

      template <stdexec::__decays_to<__t> Self, class Env>
      friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
        -> completion_signatures<Self, Env>
        requires true;

      friend auto tag_invoke(stdexec::get_env_t, const __t& self) //
        noexcept(stdexec::__nothrow_callable<stdexec::get_env_t, const Sender&>)
          -> stdexec::__call_result_t<stdexec::get_env_t, const Sender&> {
        return stdexec::get_env(self.sndr_);
      }
    };
  };

  namespace multi_gpu_bulk {
    template <int BlockThreads, class... As, std::integral Shape, class Fun>
    __launch_bounds__(BlockThreads) __global__
      void kernel(Shape begin, Shape end, Fun fn, As... as) {
      static_assert(trivially_copyable<Shape, Fun, As...>);
      const Shape i = begin + static_cast<Shape>(threadIdx.x + blockIdx.x * blockDim.x);

      if (i < end) {
        ::cuda::std::move(fn)(i, ::cuda::std::forward<As>(as)...);
      }
    }

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t;

    template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
    struct receiver_t {
      using Receiver = stdexec::__t<ReceiverId>;

      class __t : public stream_receiver_base {
        Shape shape_;
        Fun f_;

        operation_t<CvrefSenderId, ReceiverId, Shape, Fun>& op_state_;

        static std::pair<Shape, Shape>
          even_share(Shape n, std::size_t rank, std::size_t size) noexcept {
          const auto avg_per_thread = n / size;
          const auto n_big_share = avg_per_thread + 1;
          const auto big_shares = n % size;
          const auto is_big_share = rank < big_shares;
          const auto begin = is_big_share
                             ? n_big_share * rank
                             : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
          const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

          return std::make_pair(begin, end);
        }

       public:
        using __id = receiver_t;

        template <class... As>
          requires __callable<Fun, Shape, As&...>
        void set_value(As&&... as) noexcept {
          // TODO Manage errors
          // TODO Usual logic when there's only a single GPU
          cudaStream_t baseline_stream = op_state_.get_stream();
          cudaEventRecord(op_state_.ready_to_launch_, baseline_stream);

          if (shape_) {
            constexpr int block_threads = 256;
            for (int dev = 0; dev < op_state_.num_devices_; dev++) {
              if (op_state_.current_device_ != dev) {
                cudaStream_t stream = op_state_.streams_[dev];
                auto [begin, end] = even_share(shape_, dev, op_state_.num_devices_);
                auto shape = static_cast<int>(end - begin);
                const int grid_blocks = (shape + block_threads - 1) / block_threads;

                if (begin < end) {
                  cudaSetDevice(dev);
                  cudaStreamWaitEvent(stream, op_state_.ready_to_launch_, 0);
                  kernel<block_threads, As&...>
                    <<<grid_blocks, block_threads, 0, stream>>>(begin, end, f_, as...);
                  cudaEventRecord(op_state_.ready_to_complete_[dev], op_state_.streams_[dev]);
                }
              }
            }

            {
              const int dev = op_state_.current_device_;
              cudaSetDevice(dev);
              auto [begin, end] = even_share(shape_, dev, op_state_.num_devices_);
              auto shape = static_cast<int>(end - begin);
              const int grid_blocks = (shape + block_threads - 1) / block_threads;

              if (begin < end) {
                kernel<block_threads, As&...>
                  <<<grid_blocks, block_threads, 0, baseline_stream>>>(begin, end, f_, as...);
              }
            }

            for (int dev = 0; dev < op_state_.num_devices_; dev++) {
              if (dev != op_state_.current_device_) {
                cudaStreamWaitEvent(baseline_stream, op_state_.ready_to_complete_[dev], 0);
              }
            }
          }

          if (cudaError_t status = STDEXEC_DBG_ERR(cudaPeekAtLastError()); status == cudaSuccess) {
            op_state_.propagate_completion_signal(stdexec::set_value, static_cast<As&&>(as)...);
          } else {
            op_state_.propagate_completion_signal(stdexec::set_error, std::move(status));
          }
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          op_state_.propagate_completion_signal(set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          op_state_.propagate_completion_signal(set_stopped_t());
        }

        auto get_env() const noexcept -> env_of_t<Receiver> {
          return stdexec::get_env(op_state_.rcvr_);
        }

        explicit __t(
          Shape shape,
          Fun fun,
          operation_t<CvrefSenderId, ReceiverId, Shape, Fun>& op_state)
          : shape_(shape)
          , f_(static_cast<Fun&&>(fun))
          , op_state_(op_state) {
        }
      };
    };

    template <class SenderId, class ReceiverId, class Shape, class Fun>
    using operation_base_t =
      operation_state_t<SenderId, receiver_t<SenderId, ReceiverId, Shape, Fun>, ReceiverId>;

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun>
    struct operation_t : operation_base_t<CvrefSenderId, ReceiverId, Shape, Fun> {
      using Sender = __cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;

      template <class _Receiver2>
      operation_t(
        int num_devices,
        Sender&& __sndr,
        _Receiver2&& __rcvr,
        Shape shape,
        Fun fun,
        context_state_t context_state)
        : operation_base_t<CvrefSenderId, ReceiverId, Shape, Fun>(
          static_cast<Sender&&>(__sndr),
          static_cast<_Receiver2&&>(__rcvr),
          [&](operation_state_base_t<stdexec::__id<_Receiver2>>&)
            -> stdexec::__t<receiver_t<CvrefSenderId, ReceiverId, Shape, Fun>> {
            return stdexec::__t<receiver_t<CvrefSenderId, ReceiverId, Shape, Fun>>(
              shape, fun, *this);
          },
          context_state)
        , num_devices_(num_devices)
        , streams_(new cudaStream_t[num_devices_])
        , ready_to_complete_(new cudaEvent_t[num_devices_]) {
        // TODO Manage errors
        cudaGetDevice(&current_device_);
        cudaEventCreate(&ready_to_launch_, cudaEventDisableTiming);
        for (int dev = 0; dev < num_devices_; dev++) {
          cudaSetDevice(dev);
          cudaStreamCreate(streams_.get() + dev);
          cudaEventCreate(ready_to_complete_.get() + dev, cudaEventDisableTiming);
        }
        cudaSetDevice(current_device_);
      }

      ~operation_t() {
        // TODO Manage errors
        for (int dev = 0; dev < num_devices_; dev++) {
          cudaSetDevice(dev);
          cudaStreamDestroy(streams_[dev]);
          cudaEventDestroy(ready_to_complete_[dev]);
        }
        cudaSetDevice(current_device_);
        cudaEventDestroy(ready_to_launch_);
      }

      STDEXEC_IMMOVABLE(operation_t);

      int num_devices_{};
      int current_device_{};
      std::unique_ptr<cudaStream_t[]> streams_;
      std::unique_ptr<cudaEvent_t[]> ready_to_complete_;
      cudaEvent_t ready_to_launch_;
    };
  } // namespace multi_gpu_bulk

  template <class SenderId, std::integral Shape, class Fun>
  struct multi_gpu_bulk_sender_t {
    using sender_concept = stdexec::sender_t;
    using Sender = stdexec::__t<SenderId>;

    struct __t : stream_sender_base {
      using __id = multi_gpu_bulk_sender_t;
      int num_devices_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      using _set_error_t = completion_signatures<set_error_t(cudaError_t)>;

      template <class... Tys>
      using _set_value_t = completion_signatures<set_value_t(Tys...)>;

      template <class Self, class Env>
      using _completion_signatures_t = //
        __try_make_completion_signatures<
          __copy_cvref_t<Self, Sender>,
          Env,
          _set_error_t,
          __q<_set_value_t>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, _completion_signatures_t<Self, env_of_t<Receiver>>>
      STDEXEC_MEMFN_DECL(
        auto connect)(this Self&& self, Receiver&& rcvr) //
        -> multi_gpu_bulk::operation_t<__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun> {
        auto sch = stdexec::get_completion_scheduler<set_value_t>(stdexec::get_env(self.sndr_));
        context_state_t context_state = sch.context_state_;
        return multi_gpu_bulk::
          operation_t<__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun>(
            self.num_devices_,
            static_cast<Self&&>(self).sndr_,
            static_cast<Receiver&&>(rcvr),
            self.shape_,
            self.fun_,
            context_state);
      }

      template <__decays_to<__t> Self, class Env>
      static auto get_completion_signatures(Self&&, Env&&) -> _completion_signatures_t<Self, Env> {
        return {};
      }

      auto get_env() const noexcept -> env_of_t<const Sender&> {
        return stdexec::get_env(sndr_);
      }
    };
  };
} // namespace nvexec::STDEXEC_STREAM_DETAIL_NS

namespace stdexec::__detail {
  template <class SenderId, class Shape, class Fun>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::bulk_sender_t<__name_of<__t<SenderId>>, Shape, Fun>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::bulk_sender_t<SenderId, Shape, Fun>>{};

  template <class SenderId, class Shape, class Fun>
  inline constexpr __mconst<
    nvexec::STDEXEC_STREAM_DETAIL_NS::multi_gpu_bulk_sender_t<__name_of<__t<SenderId>>, Shape, Fun>>
    __name_of_v<nvexec::STDEXEC_STREAM_DETAIL_NS::multi_gpu_bulk_sender_t<SenderId, Shape, Fun>>{};
} // namespace stdexec::__detail
