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
//! \\file {{className}}.cpp
//! \\ingroup lbm
//! \\author lbmpy
//======================================================================================================================

#include <cmath>

#include "core/DataTypes.h"
#include "core/Macros.h"
#include "{{class_name}}.h"


{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#endif

using namespace std;

namespace walberla {
namespace {{namespace}} {

{{kernel|generate_definition(target)}}

void {{class_name}}::operator() ( IBlock * block{%if target is equalto 'gpu'%} , cudaStream_t stream{% endif %} )
{
    {{kernel|generate_block_data_to_field_extraction|indent(4)}}
    {{kernel|generate_call(stream='stream')|indent(4)}}
    {{kernel|generate_swaps|indent(4)}}
}



void {{class_name}}::inner( IBlock * block{%if target is equalto 'gpu'%} , cudaStream_t stream{% endif %} )
{
    {{kernel|generate_block_data_to_field_extraction|indent(4)}}

    CellInterval inner = {{field}}->xyzSize();
    inner.expand(-1);

    {{kernel|generate_call(stream='stream', cell_interval='inner')|indent(4)}}
}


void {{class_name}}::outer( IBlock * block{%if target is equalto 'gpu'%} , cudaStream_t stream {% endif %} )
{
    static std::vector<CellInterval> layers;

    {{kernel|generate_block_data_to_field_extraction|indent(4)}}

    if( layers.size() == 0 )
    {
        CellInterval ci;

        {{field}}->getSliceBeforeGhostLayer(stencil::T, ci, 1, false);
        layers.push_back(ci);
        {{field}}->getSliceBeforeGhostLayer(stencil::B, ci, 1, false);
        layers.push_back(ci);

        {{field}}->getSliceBeforeGhostLayer(stencil::N, ci, 1, false);
        ci.expand(Cell(0, 0, -1));
        layers.push_back(ci);
        {{field}}->getSliceBeforeGhostLayer(stencil::S, ci, 1, false);
        ci.expand(Cell(0, 0, -1));
        layers.push_back(ci);

        {{field}}->getSliceBeforeGhostLayer(stencil::E, ci, 1, false);
        ci.expand(Cell(0, -1, -1));
        layers.push_back(ci);
        {{field}}->getSliceBeforeGhostLayer(stencil::W, ci, 1, false);
        ci.expand(Cell(0, -1, -1));
        layers.push_back(ci);
    }

    {%if target is equalto 'gpu'%}
    {
        auto parallelSection_ = parallelStreams_.parallelSection( stream );
        for( auto & ci: layers )
        {
            parallelSection_.run([&]( auto s ) {
                {{kernel|generate_call(stream='s', cell_interval='ci')|indent(16)}}
            });
        }
    }
    {% else %}
    for( auto & ci: layers )
    {
        {{kernel|generate_call(cell_interval='ci')|indent(8)}}
    }
    {% endif %}

    {{kernel|generate_swaps|indent(4)}}
}


} // namespace {{namespace}}
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif
