



class {{modelName}}
{
   typedef stencil::{{stencilName}} Stencil;
   typedef stencil::{{stencilName}} CommunicationStencil;
   static const real_t w[{{Q}}];
   static const real_t wInv[{{Q}}];

   static const bool compressible = {{compressible}};
   static const int equilibriumAccuracyOrder = {{equilibriumAccuracyOrder}};

   //TODO packing/unpacking and configure
   void pack( mpi::SendBuffer & buffer) const {}
   void unpack( mpi::RecvBuffer & buffer) {}
   void configure( IBlock & block, StructuredBlockStorage & sbs) {

   }

private:
   {% for relaxationRate in relaxationRates -%}
   real_t {{relaxationRate}};
   {% endfor -%}

   {% for forceComponent in forceParameters %}
   real_t {{forceComponent}};
   {% endfor %}
};

const real_t  {{modelName}}::w[19] = { {{weights}} }
const real_t  {{modelName}}::wInv[19] = { {{inverseWeights}} };


// EquilibriumDistribution

template<>
class EquilibriumDistribution<MyLatticeModel, void>
{
   typedef typename LatticeModel_T::Stencil Stencil;


   static real_t get( const real_t cx, const real_t cy, const real_t cz, const real_t w,
                      const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ),
                      const real_t rho = real_t(1.0) )
   {
   }

   static real_t get( const stencil::Direction direction,
                      const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ),
                      const real_t rho = real_t(1.0) )
   {
      using namespace stencil;
      return get( real_c(cx[direction]), real_c(cy[direction]), real_c(cz[direction]),
                  LatticeModel_T::w[ Stencil::idx[direction] ], velocity, rho );
   }

   static real_t getSymmetricPart( const stencil::Direction direction,
                                   const Vector3<real_t> & velocity = Vector3< real_t >(real_t(0.0)),
                                   const real_t rho = real_t(1.0) )
   {
   }

   static real_t getAsymmetricPart( const stencil::Direction direction,
                                    const Vector3< real_t > & velocity = Vector3<real_t>( real_t(0.0) ),
                                    const real_t rho = real_t(1.0) )
   {
   }

   static std::vector< real_t > get( const Vector3< real_t > & velocity = Vector3<real_t>( real_t(0.0) ),
                                     const real_t rho = real_t(1.0) )
   {
      std::vector< real_t > equilibrium( Stencil::Size );

      return equilibrium;
   }
};



template<>
struct AdaptVelocityToForce<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static Vector3<real_t> get( FieldPtrOrIterator & it, const LatticeModel_T & latticeModel, const Vector3< real_t > & velocity, const real_t )
   {
      return velocity - latticeModel.forceModel().force(it.x(),it.y(),it.z()) * real_t(0.5);
   }

   static Vector3<real_t> get( const cell_idx_t x, const cell_idx_t y, const cell_idx_t z, const LatticeModel_T & latticeModel,
                               const Vector3< real_t > & velocity, const real_t )
   {
      return velocity - latticeModel.forceModel().force(x,y,z) * real_t(0.5);
   }
};

template<>
struct Equilibrium< MyLatticeModel, void >
{

   template< typename FieldPtrOrIterator >
   static void set( FieldPtrOrIterator & it,
                    const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
   }

   template< typename PdfField_T >
   static void set( PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z,
                    const Vector3< real_t > & velocity = Vector3< real_t >( real_t(0.0) ), const real_t rho = real_t(1.0) )
   {
   }
};


template<>
struct Density<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static inline real_t get( const LatticeModel_T & , const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static inline real_t get( const LatticeModel_T & ,
                             const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }
};


template<>
struct DensityAndMomentumDensity<MyLatticeModel, void>
{
   template< typename FieldPtrOrIterator >
   static real_t getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & ,
                                 const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static real_t getEquilibrium( Vector3< real_t > & momentumDensity, const LatticeModel_T & ,
                                 const PdfField_T & pdf, const cell_idx_t x, const cell_idx_t y, const cell_idx_t z )
   {
   }

   template< typename FieldPtrOrIterator >
   static real_t get( Vector3< real_t > & momentumDensity, const LatticeModel_T & ,
                      const FieldPtrOrIterator & it )
   {
   }

   template< typename PdfField_T >
   static real_t get( Vector3< real_t > & momentumDensity, const LatticeModel_T & , const PdfField_T & pdf,
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

