/*
 * Copyright (c) 2023 ETH Zurich
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

#include "../stdexec/execution.hpp"
#include "inline_scheduler.hpp"

#include <array>
#include <cstddef>

namespace exec {
namespace __bulk_nested {
using namespace stdexec;

template <class _ReceiverId, integral _Shape, std::size_t N, class _Fun>
struct __receiver {
  using _Receiver = stdexec::__t<_ReceiverId>;

  class __t : receiver_adaptor<__t, _Receiver> {
    friend receiver_adaptor<__t, _Receiver>;

    [[no_unique_address]] std::array<_Shape, N> __shape_;
    [[no_unique_address]] _Fun __f_;

    template <class... _As>
    void set_value(_As &&...__as) &&noexcept requires
        __nothrow_callable<_Fun, inline_scheduler, _Shape, _As &...> {
      // TODO: Propagate nested dimensions to inline scheduler? Currently
      // everything but the first dimension is dropped, but maybe it doesn't
      // matter in the default implementation?
      for (_Shape __i{}; __i != __shape_[0]; ++__i) {
        __f_(inline_scheduler{}, __i, __as...);
      }
      stdexec::set_value(std::move(this->base()), (_As &&) __as...);
    }

    template <class... _As>
    void set_value(_As &&...__as) &&noexcept requires
        __callable<_Fun, inline_scheduler, _Shape, _As &...> {
      try {
        // TODO: Propagate nested dimensions to inline scheduler? Currently
        // everything but the first dimension is dropped, but maybe it doesn't
        // matter in the default implementation?
        for (_Shape __i{}; __i != __shape_[0]; ++__i) {
          __f_(inline_scheduler{}, __i, __as...);
        }
        stdexec::set_value(std::move(this->base()), (_As &&) __as...);
      } catch (...) {
        stdexec::set_error(std::move(this->base()), std::current_exception());
      }
    }

  public:
    using __id = __receiver;

    explicit __t(_Receiver __rcvr, std::array<_Shape, N> __shape, _Fun __fun)
        : receiver_adaptor<__t, _Receiver>((_Receiver &&) __rcvr),
          __shape_(__shape), __f_((_Fun &&) __fun) {}
  };
};

template <class _Ty> using __decay_ref = __decay_t<_Ty> &;

template <class _SenderId, integral _Shape, std::size_t N, class _Fun>
struct __sender {
  using _Sender = stdexec::__t<_SenderId>;

  template <receiver _Receiver>
  using __receiver =
      stdexec::__t<__receiver<stdexec::__id<_Receiver>, _Shape, N, _Fun>>;

  struct __t {
    using __id = __sender;
    using is_sender = void;

    [[no_unique_address]] _Sender __sndr_;
    [[no_unique_address]] std::array<_Shape, N> __shape_;
    [[no_unique_address]] _Fun __fun_;

    template <class _Sender, class _Env>
    using __with_error_invoke_t = //
        __if_c<__v<__value_types_of_t<
                   _Sender, _Env,
                   __transform<__q<__decay_ref>,
                               __mbind_front_q<__non_throwing_, _Fun,
                                               inline_scheduler, _Shape>>,
                   __q<__mand>>>,
               completion_signatures<>, __with_exception_ptr>;

    template <class _Self, class _Env>
    using __completion_signatures = //
        __make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>, _Env,
            __with_error_invoke_t<__copy_cvref_t<_Self, _Sender>, _Env>>;

    template <__decays_to<__t> _Self, receiver _Receiver>
    requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
    friend auto tag_invoke(connect_t, _Self &&__self, _Receiver __rcvr) //
        noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _Sender>,
                                       __receiver<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>,
                                __receiver<_Receiver>> {
      return stdexec::connect(
          ((_Self &&) __self).__sndr_,
          __receiver<_Receiver>{(_Receiver &&) __rcvr, __self.__shape_,
                                ((_Self &&) __self).__fun_});
    }

    template <__decays_to<__t> _Self, class _Env>
    friend auto tag_invoke(get_completion_signatures_t, _Self &&, _Env)
        -> dependent_completion_signatures<_Env>;

    template <__decays_to<__t> _Self, class _Env>
    friend auto tag_invoke(get_completion_signatures_t, _Self &&, _Env)
        -> __completion_signatures<_Self, _Env>
    requires true;

    friend auto tag_invoke(get_env_t, const __t &__self) //
        noexcept(__nothrow_callable<get_env_t, const _Sender &>)
            -> __call_result_t<get_env_t, const _Sender &> {
      return get_env(__self.__sndr_);
    }
  };
};

struct bulk_nested_t {
  template <sender _Sender, integral _Shape, std::size_t N, class _Fun>
  using __sender =
      __t<__sender<stdexec::__id<__decay_t<_Sender>>, _Shape, N, _Fun>>;

  template <sender _Sender, integral _Shape, std::size_t N,
            __movable_value _Fun>
  requires __tag_invocable_with_completion_scheduler<
      bulk_nested_t, set_value_t, _Sender, std::array<_Shape, N>, _Fun>
      sender auto operator()(_Sender &&__sndr, std::array<_Shape, N> __shape,
                             _Fun __fun) const
      noexcept(nothrow_tag_invocable<
               bulk_nested_t, __completion_scheduler_for<_Sender, set_value_t>,
               _Sender, std::array<_Shape, N>, _Fun>) {
    auto __sched = get_completion_scheduler<set_value_t>(get_env(__sndr));
    return tag_invoke(bulk_nested_t{}, std::move(__sched), (_Sender &&) __sndr,
                      __shape, (_Fun &&) __fun);
  }

  template <sender _Sender, integral _Shape, std::size_t N,
            __movable_value _Fun>
  requires(!__tag_invocable_with_completion_scheduler<
           bulk_nested_t, set_value_t, _Sender, std::array<_Shape, N>, _Fun>) &&
      tag_invocable<bulk_nested_t, _Sender, std::array<_Shape, N>, _Fun> sender
      auto
      operator()(_Sender &&__sndr, std::array<_Shape, N> __shape,
                 _Fun __fun) const
      noexcept(nothrow_tag_invocable<bulk_nested_t, _Sender,
                                     std::array<_Shape, N>, _Fun>) {
    return tag_invoke(bulk_nested_t{}, (_Sender &&) __sndr, __shape,
                      (_Fun &&) __fun);
  }

  template <sender _Sender, integral _Shape, std::size_t N,
            __movable_value _Fun>
  requires(!__tag_invocable_with_completion_scheduler<
           bulk_nested_t, set_value_t, _Sender, std::array<_Shape, N>, _Fun>) &&
      (!tag_invocable<bulk_nested_t, _Sender, std::array<_Shape, N>, _Fun>)
          __sender<_Sender, _Shape, N, _Fun>
          operator()(_Sender &&__sndr, std::array<_Shape, N> __shape,
                     _Fun __fun) const {
    return __sender<_Sender, _Shape, N, _Fun>{(_Sender &&) __sndr, __shape,
                                              (_Fun &&) __fun};
  }

  template <integral _Shape, std::size_t N, class _Fun>
  __binder_back<bulk_nested_t, std::array<_Shape, N>, _Fun>
  operator()(std::array<_Shape, N> __shape, _Fun __fun) const {
    return {{}, {}, {__shape, (_Fun &&) __fun}};
  }
};
} // namespace __bulk_nested

using __bulk_nested::bulk_nested_t;
inline constexpr bulk_nested_t bulk_nested{};
} // namespace exec
  // namespace exec
