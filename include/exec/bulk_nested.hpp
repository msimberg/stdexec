/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "../stdexec/__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__diagnostics.hpp"
#include "../stdexec/__detail/__domain.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__senders_core.hpp"
#include "../stdexec/__detail/__sender_adaptor_closure.hpp"
#include "../stdexec/__detail/__transform_completion_signatures.hpp"
#include "../stdexec/__detail/__transform_sender.hpp"

#include "inline_scheduler.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.bulk_nested]
  namespace __bulk_nested {
    using namespace stdexec;
    using namespace stdexec::__detail;
    inline constexpr __mstring __bulk_nested_context = "In exec::bulk_nested(Sender, Shape, Function)..."_mstr;
    using __on_not_callable = __callable_error<__bulk_nested_context>;

    template <class _Shape, class _Fun>
    struct __data {
      _Shape __shape_;
      STDEXEC_ATTRIBUTE((no_unique_address))
      _Fun __fun_;
      static constexpr auto __mbrs_ = __mliterals<&__data::__shape_, &__data::__fun_>();
    };
    template <class _Shape, class _Fun>
    __data(_Shape, _Fun) -> __data<_Shape, _Fun>;

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _CvrefSender, class _Env, class _Shape, class _Fun, class _Catch>
    using __with_error_invoke_t = //
      __if<
        __value_types_of_t<
          _CvrefSender,
          _Env,
          __transform<
            __q<__decay_ref>,
            __mbind_front<__mtry_catch_q<__nothrow_invocable_t, _Catch>, _Fun, inline_scheduler, typename _Shape::value_type>>,
          __q<__mand>>,
        completion_signatures<>,
        __eptr_completion>;

    template <class _CvrefSender, class _Env, class _Shape, class _Fun>
    using __completion_signatures = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __with_error_invoke_t<_CvrefSender, _Env, _Shape, _Fun, __on_not_callable>>;

    struct bulk_nested_t {
      template <sender _Sender, class _Shape, __movable_value _Fun>
      STDEXEC_ATTRIBUTE((host, device))
      auto
        operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const -> __well_formed_sender
        auto {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<bulk_nested_t>(
            __data{__shape, static_cast<_Fun&&>(__fun)}, static_cast<_Sender&&>(__sndr)));
      }

      template <class _Shape, class _Fun>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        operator()(_Shape __shape, _Fun __fun) const -> __binder_back<bulk_nested_t, _Shape, _Fun> {
        return {
          {static_cast<_Shape&&>(__shape), static_cast<_Fun&&>(__fun)},
          {},
          {}
        };
      }

      // This describes how to use the pieces of a bulk_nested sender to find
      // legacy customizations of the bulk_nested algorithm.
      using _Sender = __1;
      using _Shape = __nth_member<0>(__0);
      using _Fun = __nth_member<1>(__0);
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          bulk_nested_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Shape,
          _Fun),
        tag_invoke_t(bulk_nested_t, _Sender, _Shape, _Fun)>;
    };

    struct __bulk_nested_impl : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, _Env&&) noexcept
        -> __completion_signatures<__child_of<_Sender>, _Env, __shape_t<_Sender>, __fun_t<_Sender>> {
        static_assert(sender_expr_for<_Sender, bulk_nested_t>);
        return {};
      };

      static constexpr auto complete = //
        []<class _Tag, class _State, class _Receiver, class... _Args>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          using __shape_t = decltype(__state.__shape_)::value_type;
          if constexpr (noexcept(__state.__fun_(inline_scheduler{}, __shape_t{}, __args...))) {
            for (__shape_t __i{}; __i != __state.__shape_[0]; ++__i) {
              __state.__fun_(inline_scheduler{}, __i, __args...);
            }
            _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
          } else {
            try {
              for (__shape_t __i{}; __i != __state.__shape_[0]; ++__i) {
                __state.__fun_(inline_scheduler{}, __i, __args...);
              }
              _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
            } catch (...) {
              stdexec::set_error(static_cast<_Receiver&&>(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
        }
      };
    };
  } // namespace __bulk_nested

  using __bulk_nested::bulk_nested_t;
  inline constexpr bulk_nested_t bulk_nested{};
} // namespace exec

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::bulk_nested_t> : exec::__bulk_nested::__bulk_nested_impl { };
} // namespace exec

STDEXEC_PRAGMA_POP()
