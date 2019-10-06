#pragma once

#include "core/debug/Debug.h"
#include "communication/UniformMPIDatatypeInfo.h"
#include "field/communication/MPIDatatypes.h"

#include <set>

namespace walberla {
namespace {{namespace}} {

class {{class_name}} : public ::walberla::communication::UniformMPIDatatypeInfo
{
public:
    using GhostLayerField_T = GhostLayerField<real_t, {{f_size}}>;

    {{class_name}}( BlockDataID {{field_name}} )
        :{{field_name}}_({{field_name}})
    {}
    virtual ~{{class_name}}() {}

    virtual shared_ptr<mpi::Datatype> getSendDatatype ( IBlock * block, const stencil::Direction dir )
    {
       {% if kind == 'pull' %}
        return make_shared<mpi::Datatype>( field::communication::mpiDatatypeSliceBeforeGhostlayerXYZ(
                *getField( block ), dir, uint_t( 1 ), getOptimizedCommunicationIndices( dir ), false ) );
       {% else %}
        return make_shared<mpi::Datatype>( field::communication::mpiDatatypeGhostLayerOnlyXYZ(
                *getField( block ), dir, false, getOptimizedCommunicationIndices( dir ) ) );
       {% endif %}
    }

    virtual shared_ptr<mpi::Datatype> getRecvDatatype ( IBlock * block, const stencil::Direction dir )
    {
        {% if kind == 'pull' %}
        return make_shared<mpi::Datatype>( field::communication::mpiDatatypeGhostLayerOnlyXYZ(
                *getField( block ), dir, false, getOptimizedCommunicationIndices( stencil::inverseDir[dir] ) ) );
        {% else %}
        return make_shared<mpi::Datatype>( field::communication::mpiDatatypeSliceBeforeGhostlayerXYZ(
                *getField( block ), dir, uint_t( 1 ), getOptimizedCommunicationIndices( stencil::inverseDir[dir] ), false ) );
        {% endif %}
    }

    virtual void * getSendPointer( IBlock * block, const stencil::Direction ) {
        return getField(block)->data();
    }

    virtual void * getRecvPointer( IBlock * block, const stencil::Direction ) {
        return getField(block)->data();
    }

private:

    inline static std::set< cell_idx_t > getOptimizedCommunicationIndices( const stencil::Direction dir )
    {
        switch(dir)
        {
            {%- for direction_set, index_set in spec.items()  %}
            {%- for dir in direction_set %}
            case stencil::{{dir}}:
            {%- endfor %}
               return {{index_set}};
            {% endfor %}
            default:
                WALBERLA_ASSERT(false);
                return {};
        }
    }

    GhostLayerField_T * getField( IBlock * block )
    {
        GhostLayerField_T * const f = block->getData<GhostLayerField_T>( {{field_name}}_ );
        WALBERLA_ASSERT_NOT_NULLPTR( f );
        return f;
    }

    BlockDataID {{field_name}}_;
};


} // namespace {{namespace}}
} // namespace walberla


