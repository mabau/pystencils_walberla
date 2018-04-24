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


namespace walberla {
namespace {{namespace}} {


class {{class_name}}
{
public:
    {{class_name}}( {{kernel|generate_constructor_parameters}} )
        : {{ kernel|generate_constructor_initializer_list }}
    {};

    void operator() ( IBlock * block );

private:
    {{kernel|generate_members|indent(4)}}
};


} // namespace {{namespace}}
} // namespace walberla
