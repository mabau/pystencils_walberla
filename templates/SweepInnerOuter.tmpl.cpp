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

{{kernel|generate_definition}}

{% for outer_kernel in outer_kernels.values() %}
{{outer_kernel|generate_definition}}
{% endfor %}


void {{class_name}}::operator() ( IBlock * block )
{
    {{kernel|generate_block_data_to_field_extraction|indent(4)}}
    {{kernel|generate_call(stream='stream_')|indent(4)}}
    {{kernel|generate_swaps|indent(4)}}
}



void {{class_name}}::inner( IBlock * block )
{
    {{kernel|generate_block_data_to_field_extraction|indent(4)}}

    CellInterval inner = {{field}}->xyzSize();
    inner.expand(-1);

    {{kernel|generate_call(stream='stream_', cell_interval='inner')|indent(4)}}
}


void {{class_name}}::outer( IBlock * block )
{
    static std::vector<CellInterval> layers;
    {%if target is equalto 'gpu'%}
    static std::vector<cudaStream_t> streams;
    {% endif %}

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

        {%if target is equalto 'gpu'%}
        for( int i=0; i < layers.size(); ++i )
        {
            streams.push_back(cudaStream_t());
            WALBERLA_CUDA_CHECK( cudaStreamCreateWithPriority(&streams.back(), cudaStreamDefault, -1) );
        }
        {% endif %}
    }

    { {{outer_kernels['W']|generate_call(stream='streams[5]', cell_interval="layers[5]")|indent(4)}} }
    { {{outer_kernels['E']|generate_call(stream='streams[4]', cell_interval="layers[4]")|indent(4)}} }
    { {{outer_kernels['S']|generate_call(stream='streams[3]', cell_interval="layers[3]")|indent(4)}} }
    { {{outer_kernels['N']|generate_call(stream='streams[2]', cell_interval="layers[2]")|indent(4)}} }
    { {{outer_kernels['B']|generate_call(stream='streams[1]', cell_interval="layers[1]")|indent(4)}} }
    { {{outer_kernels['T']|generate_call(stream='streams[0]', cell_interval="layers[0]")|indent(4)}} }

    for(int i=0; i < layers.size(); ++i )
        WALBERLA_CUDA_CHECK( cudaStreamSynchronize(streams[i]) );

    {{kernel|generate_swaps|indent(4)}}
}


} // namespace {{namespace}}
} // namespace walberla


#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic pop
#endif
