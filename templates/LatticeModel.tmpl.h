

class AbstractLbmModel
{
      virtual void stream       ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) ) = 0;
      virtual void collide      ( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) ) = 0;
      virtual void streamCollide( IBlock * const block, const uint_t numberOfGhostLayersToInclude = uint_t(0) ) = 0;
};


class {{className}} : public AbstractLbmModel
{
public:
   typedef stencil::{{stencilName}} Stencil;
   typedef stencil::{{stencilName}} CommunicationStencil;
   static const real_t w[{{Q}}];
   static const real_t wInv[{{Q}}];

   static const bool compressible = {{compressible}};
   static const int equilibriumAccuracyOrder = {{equilibriumAccuracyOrder}};


   {{className}}( {{streamCollideKernel|generateConstructorParameters}} )
       : {{ streamCollideKernel|generateConstructorInitializerList }}
   {};


   //TODO
   void pack( mpi::SendBuffer & buffer) const {
   }
   void unpack( mpi::RecvBuffer & buffer) {
   }
   void configure( IBlock & block, StructuredBlockStorage & sbs) {
       {{streamCollideKernel|generateBlockDataToFieldExtraction(['pdfs', 'pdfs_tmp'])|indent(8)}}
   }

private:
    {{streamCollideKernel|generateMembers|indent(4)}}

};

const real_t  {{className}}::w[19] = { {{weights}} }c
const real_t  {{className}}::wInv[19] = { {{inverseWeights}} };


// EquilibriumDistribution

template<>
class EquilibriumDistribution<MyLatticeModel, void>
{
   typedef typename LatticeModel_T::Stencil Stencil;

   /*
   static real_t get( const real_t cx, const real_t cy, const real_t cz, const real_t w,
                      const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ),
                      const real_t rho = real_t(1.0) )
   */

   static real_t get( const stencil::Direction direction,
                      const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ),
                      const real_t rho = real_t(1.0) )
   {
        {{equilibriumFromDirection}}
   }

   static real_t getSymmetricPart( const stencil::Direction direction,
                                   const Vector3<real_t> & u = Vector3< real_t >(real_t(0.0)),
                                   const real_t rho = real_t(1.0) )
   {
        {{symmetricEquilibriumFromDirection}}
   }

   static real_t getAsymmetricPart( const stencil::Direction direction,
                                    const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                    const real_t rho = real_t(1.0) )
   {
        {{asymmetricEquilibriumFromDirection}}
   }

   static std::vector< real_t > get( const Vector3< real_t > & u = Vector3<real_t>( real_t(0.0) ),
                                     const real_t rho = real_t(1.0) )
   {
      std::vector< real_t > equilibrium( Stencil::Size );
      for( auto d = Stencil::begin(); d != Stencil::end(); ++d )
      {
         equilibrium[d.toIdx()] = getEquilibrium(*d, u, rho);
      }
      return equilibrium;
   }
};


template<>
struct AdaptVelocityToForce<MyLatticeModel, void>
{
   //TODO support force fields
   template< typename FieldPtrOrIterator >
   static Vector3<real_t> get( FieldPtrOrIterator & it, const LatticeModel_T & latticeModel,
                               const Vector3< real_t > & velocity, const real_t rho)
   {
      auto x = it.x();
      auto y = it.y();
      auto z = it.z();
      {% if macroscopicVelocityShift %}
      return velocity - Vector3<real_t>({{macroscopicVelocityShift | join(",") }});
      {% else %}
      return velocity;
      {% endif %}
   }

   static Vector3<real_t> get( const cell_idx_t , const cell_idx_t , const cell_idx_t , const LatticeModel_T & latticeModel,
                               const Vector3< real_t > & velocity, const real_t rho)
   {
      {% if macroscopicVelocityShift %}
      return velocity - Vector3<real_t>({{macroscopicVelocityShift | join(",") }});
      {% else %}
      return velocity;
      {% endif %}
   }
};

template<>
struct Equilibrium< MyLatticeModel, void >
{

   template< typename FieldPtrOrIterator >
   static void set( FieldPtrOrIterator & it,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
       {% for eqTerm in equilibrium -%}
          it[{{loop.index0 }}] = {{eqTerm}};
       {% endfor -%}

   }

   template< typename PdfField_T >
   static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                    const Vector3< real_t > & u = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
      real_t & xyz0 = pdf(x,y,z,0);
      {% for eqTerm in equilibrium -%}
         pdf.getF( &xyz0, {{loop.index0 }})= {{eqTerm}};
      {% endfor -%}

   }
};


template<>
struct Density<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const LatticeModel_T & , const FieldPtrOrIterator & it )
   {
        {% for i in range(Q) -%}
            const real_t f_{{i}} = it[{{i}}];
        {% endfor -%}
        {{densityOut | indent(8)}}
        return rho;
   }

   template< typename PdfField_T >
   static inline real_t get( const LatticeModel_T & ,
                             const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
        const real_t & xyz0 = pdf(x,y,z,0);
        {% for i in range(Q) -%}
            const real_t f_{{i}} = pdf.getF( &xyz0, {{i}});
        {% endfor -%}
        {{densityOut | indent(8)}}
        return rho;
   }
};



template<>
struct DensityAndMomentumDensity<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static real_t getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & lm,
                                 const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static real_t getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & lm,
                                 const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }

   template< typename FieldPtrOrIterator >
   static real_t get( Vector3< real_t > & momentumDensity, const LatticeModel_T & lm,
                      const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static real_t get( Vector3< real_t > & momentumDensity, const LatticeModel_T & lm, const PdfField_T & pdf,
                      const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }
};



template<>
struct DensityAndVelocity<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static void set( FieldPtrOrIterator & it, const LatticeModel_T & ,
                    const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
   }

   template< typename PdfField_T >
   static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const LatticeModel_T & ,
                    const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
   }
};


template<typename FieldIteratorXYZ >
struct DensityAndVelocityRange<MyLatticeModel, FieldIteratorXYZ, void>
{

   static void set( FieldIteratorXYZ & begin, const FieldIteratorXYZ & end, const LatticeModel_T & latticeModel,
                    const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
      Vector3< real_t > velAdaptedToForce = internal::AdaptVelocityToForce<LatticeModel_T>::get( latticeModel, velocity, rho );
      EquilibriumRange< LatticeModel_T, FieldIteratorXYZ >::set( begin, end, velAdaptedToForce, rho );
   }
};


template<>
struct MomentumDensity< MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static void getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & latticeModel,
                               const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static void getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & latticeModel,
                               const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }

   template< typename FieldPtrOrIterator >
   static void get( Vector3< real_t > & momentumDensity, const LatticeModel_T & latticeModel, const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static void get( Vector3< real_t > & momentumDensity, const LatticeModel_T & latticeModel, const PdfField_T & pdf,
                    const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }
};

template<>
struct PressureTensor<MyLatticeModel>
{
   template< typename FieldPtrOrIterator >
   static void get( Matrix3< real_t > & pressureTensor, const LatticeModel_T & latticeModel, const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static void get( Matrix3< real_t > & pressureTensor, const LatticeModel_T & latticeModel, const PdfField_T & pdf,
                    const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }
};


template<>
struct ShearRate<MyLatticeModel>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const LatticeModel_T & latticeModel, const FieldPtrOrIterator & it,
                             const Vector3< real_t > & velocity, const real_t rho )
   {
   }

   template< typename PdfField_T >
   static inline real_t get( const LatticeModel_T & latticeModel,
                             const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                             const Vector3< real_t > & velocity, const real_t rho )
   {
   }

   /// For incompressible LB you don't have to pass a value for 'rho' since for incompressible LB 'rho' is not used in this function!
   static inline real_t get( const std::vector< real_t > & nonEquilibrium, const real_t relaxationParam,
                             const real_t rho = real_t(1) )
   {
   }
};

