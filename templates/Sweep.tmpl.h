//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \\file {{class_name}}.h
//! \\author pystencils
//======================================================================================================================

#include "core/DataTypes.h"

{% if target is equalto 'cpu' -%}
#include "field/GhostLayerField.h"
{%- elif target is equalto 'gpu' -%}
#include "cuda/GPUField.h"
{%- endif %}
#include "field/SwapableCompare.h"
#include "domain_decomposition/BlockDataID.h"
#include "domain_decomposition/IBlock.h"

#include <set>

#ifdef __GNUC__
#define RESTRICT __restrict__
#elif _MSC_VER
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace walberla {
namespace {{namespace}} {


class {{class_name}}
{
public:
    {{class_name}}( {{kernel|generate_constructor_parameters}}{%if target is equalto 'gpu'%} , cudaStream_t stream = 0{% endif %})
        : {{ kernel|generate_constructor_initializer_list }}, stream_(stream)
    {};

    void operator() ( IBlock * block );

    void inner( IBlock * block );
    void outer( IBlock * block );

private:
    {{kernel|generate_members|indent(4)}}
    {%if target is equalto 'gpu'%}
    cudaStream_t stream_;
    {% endif %}
};


} // namespace {{namespace}}
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif
