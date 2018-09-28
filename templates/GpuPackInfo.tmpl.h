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

{% for dtype in dtypes %}
uint_t packOnGPU(stencil::Direction dir, {{dtype}} * buffer, cell_idx_t thickness,
                 GPUField<{{dtype}}> * f, cudaStream_t stream);
uint_t unpackOnGPU(stencil::Direction dir, {{dtype}} * buffer, cell_idx_t thickness,
                   GPUField<{{dtype}}> * f, cudaStream_t stream);
{% endfor %}

} // namespace cuda
} // namespace walberla
