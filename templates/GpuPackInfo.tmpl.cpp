#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "cuda/GPUField.h"
#include "core/DataTypes.h"

{% if target is equalto 'cpu' -%}
#define FUNC_PREFIX
{%- elif target is equalto 'gpu' -%}
#define FUNC_PREFIX __global__
{%- endif %}


namespace walberla {
namespace cuda {

using walberla::cell::CellInterval;
using walberla::stencil::Direction;


{% for layout in layouts %}
{% for dtype in dtypes %}

{{kernels[('pack', layout,dtype)]|generate_definition}}
{{kernels[('unpack', layout,dtype)]|generate_definition}}

{% endfor %}
{% endfor %}




{% for dtype in dtypes %}

uint_t packOnGPU(Direction dir, {{dtype}} * buffer, cell_idx_t thickness, GPUField<{{dtype}}> * f, cudaStream_t stream)
{
    CellInterval ci;
    f->getSliceBeforeGhostLayer(dir, ci, thickness, false);

    if( f->layout() == field::fzyx) {
        {{kernels[('pack', 'fzyx', dtype)]|generate_call(cell_interval="ci", stream="stream")|indent(8)}}
    } else {
        {{kernels[('pack', 'zyxf', dtype)]|generate_call(cell_interval="ci", stream="stream")|indent(8)}}
    }
    return ci.numCells();
}


uint_t unpackOnGPU(Direction dir, {{dtype}} * buffer, cell_idx_t thickness, GPUField<{{dtype}}> * f, cudaStream_t stream)
{
    CellInterval ci;
    f->getGhostRegion(dir, ci, thickness, false);

    if( f->layout() == field::fzyx) {
        {{kernels[('unpack', 'fzyx', dtype)]|generate_call(cell_interval="ci", stream="stream")|indent(8)}}
    } else {
        {{kernels[('unpack', 'zyxf', dtype)]|generate_call(cell_interval="ci", stream="stream")|indent(8)}}
    }
    return ci.numCells();
}

{% endfor %}


} // namespace cuda
} // namespace walberla