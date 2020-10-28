/*---------------------------------------------------------------------------*\
*
*  bitpit
*
*  Copyright (C) 2015-2019 OPTIMAD engineering Srl
*
*  -------------------------------------------------------------------------
*  License
*  This file is part of bitpit.
*
*  bitpit is free software: you can redistribute it and/or modify it
*  under the terms of the GNU Lesser General Public License v3 (LGPL)
*  as published by the Free Software Foundation.
*
*  bitpit is distributed in the hope that it will be useful, but WITHOUT
*  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
*  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
*  License for more details.
*
*  You should have received a copy of the GNU Lesser General Public License
*  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
*
\*---------------------------------------------------------------------------*/

#pragma once

// Standard Template Library
# include <vector>
# include <array>
# include <functional>
# include <exception>
# include <type_traits>

// Bitpit
# include "bitpit_operators.hpp"
# include "metaprogramming.hpp"


namespace bitpit{

/*!
 * @enum RBFBasisFunction
 * @ingroup RBF
 * @brief Enum class defining types of RBF kernel functions that could be used in bitpit::RBF class
 */
enum class RBFBasisFunction {
    CUSTOM     = 0,  /**<Identify custom linked support function */
    WENDLANDC2 = 1,  /**< Compact support Wendland C2 function */
    LINEAR     = 2,  /**< Compact support linear function */
    GAUSS90    = 3,  /**< Non compact gaussian with 90% of reduction at unary radius */
    GAUSS95    = 4,  /**< Non compact gaussian with 95% of reduction at unary radius */
    GAUSS99    = 5,  /**< Non compact gaussian with 99% of reduction at unary radius */
    C1C0       = 6,  /**< Compact quadratic funct, C1 on r=0, C0 on r=1, 0 outside */
    C2C0       = 7,  /**< Compact cubic funct, C2 on r=0, C0 on r=1, 0 outside */
    C0C1       = 8,  /**< Compact quadratic funct, C0 on r=0, C1 on r=1, 0 outside */
    C1C1       = 9,  /**< Compact cubic funct, C1 on r=0, C1 on r=1, 0 outside */
    C2C1       = 10, /**< Compact biquadratic funct, C2 on r=0, C1 on r=1, 0 outside */
    C0C2       = 11, /**< Compact cubic funct, C0 on r=0, C2 on r=1, 0 outside */
    C1C2       = 12, /**< Compact biquadratic funct, C1 on r=0, C2 on r=1, 0 outside */
    C2C2       = 13, /**< Compact poly (degree 5) funct, C2 on r=0, C2 on r=1, 0 outside */
};

/*!
 * @enum RBFMode
 * @ingroup RBF
 * @brief Enum class defining behaviour of the bitpit::RBF class
 */
enum class RBFMode {
    INTERP = 1, /**< RBF class interpolate external field data */
    PARAM  = 2  /**< RBF class used as pure parameterizator*/
};

class RBFKernel{

private:
    int     m_fields;                               /**<Number of data fields defined on RBF nodes.*/
    RBFMode m_mode;                                 /**<Behaviour of RBF class (interpolation or parametrization).*/
    double  m_supportRadius;                        /**<Support radius of function used as Radiabl Basis Function.*/
    RBFBasisFunction m_typef;                       /**<Recognize type of RBF shape function actually in the class. */
    double  (*m_fPtr)(double);

    std::vector<double>                 m_error;    /**<Interpolation error of a field evaluated on each RBF node (auxiliary memeber used in Greedy algorithm).*/

    protected:
    std::vector<std::vector<double>>    m_value;    /**< displ value to be interpolated on RBF nodes */
    std::vector<std::vector<double>>    m_weight;   /**< weight of your RBF interpolation */
    std::vector<bool>                   m_activeNodes;   /**<Vector of active/inactive node (m_activeNodes[i] = true/false -> the i-th node is used/not used during RBF evaluation).*/
    int m_maxFields;                                /**< fix the maximum number of fields that can be added to your class*/
    int m_nodes;                                    /**<Number of RBF nodes.*/

public:
    ~RBFKernel();
    RBFKernel();
    RBFKernel(const RBFKernel & other);

    void                    setFunction(RBFBasisFunction);
    void                    setFunction(double (&funct)(double ));

    RBFBasisFunction        getFunctionType();
    int                     getDataCount();
    int                     getActiveCount();
    std::vector<int>        getActiveSet();

    bool                    isActive(int );

    bool                    activateNode(int );
    bool                    activateNode(const std::vector<int> &);
    void                    activateAllNodes();
    bool                    deactivateNode(int );
    bool                    deactivateNode(const std::vector<int> &);
    void                    deactivateAllNodes();

    void                    setSupportRadius(double);
    double                  getSupportRadius();

    void                    setMode(RBFMode mode);
    RBFMode                 getMode();

    void                    setDataToNode (int , const std::vector<double> &);
    void                    setDataToAllNodes(int , const std::vector<double> &);

    int                     addData();
    int                     addData(const std::vector<double> &);
    bool                    removeData(int);
    bool                    removeData(std::vector<int> &);
    void                    removeAllData();

    void                    fitDataToNodes();
    void                    fitDataToNodes(int);

    std::vector<double>     evalRBF(const std::array<double,3> &);
    std::vector<double>     evalRBF(int jnode);
    double                  evalBasis(double);

    int                     solve();
    int                     greedy(double);

protected:
    double                  evalError();
    int                     addGreedyPoint();
    int                     solveLSQ();
    void                    swap(RBFKernel & x) noexcept;

private:

    virtual double calcDist(int i, int j) = 0;
    virtual double calcDist(const std::array<double,3> & point, int j) = 0;

};

class RBF : public RBFKernel {

protected:
    std::vector<std::array<double,3>>   m_node;     /**< list of RBF nodes */

public:
    ~RBF();
    RBF(RBFBasisFunction = RBFBasisFunction::WENDLANDC2);
    RBF(const RBF & other);
    RBF & operator=(RBF other);

    int                     getTotalNodesCount();

    int                     addNode(const std::array<double,3> &);
    std::vector<int>        addNode(const std::vector<std::array<double,3>> &);
    bool                    removeNode(int);
    bool                    removeNode(std::vector<int> &);
    void                    removeAllNodes();

protected:
    void     swap(RBF & x) noexcept;

private:
    double calcDist(int i, int j);
    double calcDist(const std::array<double,3> & point, int j);
};

/*!
 * @ingroup  RBF
 * @brief Utility fuctions for RBF
 */
namespace rbf
{
    double                  wendlandc2(double);
    double                  linear(double);
    double                  gauss90(double);
    double                  gauss95(double);
    double                  gauss99(double);
    double                  c1c0(double);
    double                  c2c0(double);
    double                  c0c1(double);
    double                  c1c1(double);
    double                  c2c1(double);
    double                  c0c2(double);
    double                  c1c2(double);
    double                  c2c2(double);

  /*! @addtogroup RBF
   *  @{
  */

  /*! @brief C2-continuous Wendland'2 functions.
   *
   *  C2-continuous Wendland's functions have the following expression:
   *  \f$
   *  \begin{equation}
   *      \left\{
   *        \begin{aligned}
   *          &(1 - z )^4 ( 4 z + 1 ), \; \text{if} z < 0, \\
   *          &0, \; \text{otherwise}
   *        \end{aligned}
   *    \right.
   *  \end{equation}
   *  \f$
   *  In the above expression, \f$r\f$ is the radial distance of a given point
   *  from the geometrical kernel (e.g. the center of the RF).
   *
   *  @tparam         CoordT      Type for coeffs. (only scalar floating point types
   *                              are supported, e.g. double, float, etc.)
   *
   *  @param [in]     r           radial distance.
  */
  template<
    class CoordT,
    typename std::enable_if< std::is_floating_point<CoordT>::value >::type* = nullptr >
  CoordT  wendland_c2( CoordT r )
  {
    return r > (CoordT)1 ?
      (CoordT)0 :
      std::pow( (CoordT)1 - r, 4 ) * ( (CoordT)4 * r + (CoordT)1 );
  }

  /*! @brief Generalized multiquadric functions.
   *
   *  The family of generalized multi-quadrics RF has the following expression:
   *  \f$
   *  \begin{equation}
   *    \frac{1}{ \left( c^2 + z^2 \right)^{\frac{\alpha}{\beta}} }
   *  \end{equation}
   *  \f$
   *  for \f$\alpha, \beta > 0\f$.
   *  In the above expression, \f$r\f$ is the radial distance of a given point
   *  from the kernel of the RF (e.g. the center of the RF),
   *  and \f$c\f$ is a additional parameter (bias).
   *
   *  @tparam         CoordT      type of coeffs. (e.g. double, float, etc. )
   *  @tparam         Alha, Beta  expoenent values.
   *
   *  @param [in]     r           radial distance
   *  @param [in]     c           value for the bias coeff.
  */
  template<
    class CoordT,
    unsigned Alpha,
    unsigned Beta,
    typename std::enable_if< std::is_floating_point<CoordT>::value && (Alpha > 0) && (Beta > 0) >::type* = nullptr >
  CoordT generalized_multiquadrics( CoordT r, CoordT c )
  {
    return CoordT(1)/std::pow( c*c + r*r, CoordT(Alpha)/CoordT(Beta) );
  }

  // ================================================================ //
  // FORWARD DECLARATIONS                                             //
  // ================================================================ //
  template< std::size_t, class >
  class RF;
  template< std::size_t, std::size_t, class >
  class RFP;


  /*! @brief Enum for supported families of radial functions. */
  enum eRBFType
  {
    /*! @brief undefined type. */
    // Leave it first to facilitate auto-looping through eRBFType
    kUndefined,
    /*! @brief WendLand C2-continuous radial functions. */
    kWendlandC2,
    /*! @brief Hardy's radial function (from the family of multiquadrics with \f$\alpha = 1, \beta = 2\f$)*/
    kHardy,
    /*! @brief Generalized multiquadrics with \f$\alpha = 2, \beta = 1\f$) */
    kMultiQuadric2,
    /*! @brief Generalized multiquarics with \f$\alpha = 3, \beta = 2\f$ */
    kMultiQuadric3_2,
    /*! @brief Generalized multiquarics with \f$\alpha = 5, \beta = 2\f$ */
    kMultiQuadric5_2,
    /*! @brief User-defined */
    // Leave it last to facilitate auto-looping through eRBFType.
    kUserDefined
  }; //end enum eRBFType

  // ---------------------------------------------------------------- //
  /*! @brief Helper function returning the tag associated to each type
   *  of RBF.
  */
  std::string   getRBFTag( eRBFType type );

  // ================================================================ //
  // DEFINITION OF CLASS RadialFunct									                //
  // ================================================================ //
  /*! @brief Base class used to derive radial functions of different families.
   *
   *  This class holds a radial function (RH in short) which does not depend
   *  on any additional parameter beyond its radius and its center.
   *
   *	@tparam 			Dim 			   nr. of dimension in the working space.
   *	@tparam 			CoordT 	     (default = double) type used for parameters and coordinates
   *                             (only scalar floating point type are allowed, e.g. coord_t = double, float).
  */
  template<
    std::size_t Dim,
    class CoordT = double
  >
  class RF
  {
    // Static assertions ========================================== //
    static_assert(
      (Dim > 0),
      "**ERROR** bitpit::rbf::RF<Dim,CoordT>: nr. of working dimensions, Dim, must be greater than 0"
    );
    static_assert(
      std::is_floating_point<CoordT>::value,
      "**ERROR** bitpit::rbf::RF<Dim,CoordT>: CoordT must be a integral floating point type "
      ", e.g. float, double or long double"
    );

    // Typedef(s) ================================================= //
    public:
    /*!	@brief Coeffs. type. */
    using coord_t 		= CoordT;
    /*!	@brief Point type in the working space. */
    using point_t 		= std::array<coord_t, Dim>;
    /*!	@brief Type of functor holding the actual implementation */
    using rf_funct_t  = std::function< coord_t( coord_t ) >;
    private:
    /*!	@brief Type of this object. */
    using self_t    	= RF<Dim, CoordT>;

    // Member variable(s) ========================================= //
    protected:
    /*!	@brief Function implementing the expression of the Radial Basis Function. */
    rf_funct_t        mFunct;
    /*! @brief Type of this radial function. */
    eRBFType          mType;
    /*! @brief Bool for compactly supported RFs.*/
    bool              mHasCompactSupport;
    public:
    /*!	@brief Radius of this RF. */
    coord_t           radius;
    /*!	@brief Center of this RF (sometimes referred to as control point, geometry kernel) */
    point_t           center;

    // Static member function(s) ================================== //
    public:
    /*! @brief Returns a instance of a radial function corresponding to the input
     *  type.
     *
     *  @param [in]     type        type of the RF.
     *  @param [in]     args        arguments forwarded to the constructor of the RF.
     *                              (the nr. and type of arguments depend on the specific RF)
     *
     *  @result Returns pointer to the new object.
    */
    static self_t*		New( eRBFType type )
    {
      switch( type )
      {
        default: {
          throw std::runtime_error(
            "rbf::RF::New: Undefined RF type."
          );
        } //end default
        case( bitpit::rbf::eRBFType::kWendlandC2 ): {
          auto out = new bitpit::rbf::RF<Dim, CoordT>(
            type,
            &bitpit::rbf::wendland_c2<coord_t>
          );
          out->mType = bitpit::rbf::eRBFType::kMultiQuadric2;
          out->mHasCompactSupport = true;
          return out;
        } //end case kWendlandC2
        case( bitpit::rbf::eRBFType::kHardy ): {
          auto out = new bitpit::rbf::RFP<Dim, 1, CoordT>(
            type,
            &bitpit::rbf::generalized_multiquadrics<coord_t, 1, 2>
          );
          out->mType = bitpit::rbf::eRBFType::kHardy;
          out->mHasCompactSupport = false;
          out->mParams[0] = 1;
          return out;
        } //end case kHardy
        case( bitpit::rbf::eRBFType::kMultiQuadric2 ): {
          auto out = new bitpit::rbf::RFP<Dim, 1, CoordT>(
            type,
            &bitpit::rbf::generalized_multiquadrics<coord_t, 2, 1 >
          );
          out->mType = bitpit::rbf::eRBFType::kMultiQuadric2;
          out->mHasCompactSupport = false;
          out->mParams[0] = 1;
          return out;
        } //end case kMultiQuadrics2
        case( bitpit::rbf::eRBFType::kMultiQuadric3_2 ): {
          auto out = new bitpit::rbf::RFP<Dim, 1, CoordT>(
            type,
            &bitpit::rbf::generalized_multiquadrics<coord_t, 3, 2>
          );
          out->mType = bitpit::rbf::eRBFType::kMultiQuadric3_2;
          out->mHasCompactSupport = false;
          out->mParams[0] = 1;
          return out;
        } //end case kMultiQuadric3_2
        case( bitpit::rbf::eRBFType::kMultiQuadric5_2 ): {
          auto out = new bitpit::rbf::RFP<Dim, 1, CoordT>(
            type,
            &bitpit::rbf::generalized_multiquadrics<coord_t, 5, 2>
          );
          out->mType = bitpit::rbf::eRBFType::kMultiQuadric5_2;
          out->mHasCompactSupport = false;
          out->mParams[0] = 1;
          return out;
        } //end case kMultiQuadric3_2
      } //end switch

      return nullptr;
    }

    // Member function(s) ========================================= //

    // Constructor(s) --------------------------------------------- //
    protected:
    /*! @brief Default constructor.
     *
     *  Initialize a radial function of undefined type with default values
     *  for the parameters (see #setDefault )
    */
    RF() :
      mFunct(nullptr)
      , radius()
      , center()
      , mType( bitpit::rbf::eRBFType::kUndefined )
    {
      setDefault();
    }
    public:
    /*! @brief Constructor #1.
     *
     *  Initialize a radial function with the specified expression, radius and center.
     *
     *  @param [in]     type      type of this RF.
     *  @param [in]     f         expresion of this radial function
     *  @param [in]     r         radius of this radial function.
     *  @param [in]     c         center of this radial function.
    */
    RF( bitpit::rbf::eRBFType type, coord_t (*f)(coord_t) ) :
      mFunct(f)
      , mType(type)
    {
      setDefault();
    }
    /*! @brief Constructor #2.
     *
     *  Initialize a RF object using the input functor and parameters
     *
     *  @param [in]     type      type of this RF.
     *  @param [in]     f         functor implementing the expression of this RF.
     *  @param [in]     r         radius of this radial function.
     *  @param [in]     c         center of this radial function.
    */
    RF( bitpit::rbf::eRBFType type, rf_funct_t const &f ) :
      mFunct(f)
      , mType(type)
    {
      setDefault();
    }
    /*! @brief Copy-constructor (deleted) */
    RF( const self_t & ) = delete;
    /*! @brief Move-constructor (deleted) */
    RF( self_t && ) = delete;

    // Operator(s) =============================================== //
    public:
    /*!	@brief Copy assignement operator (deleted). */
    self_t&					operator=( const self_t & ) = delete;
    /*!	@brief Move assignment operator (deleted). */
    self_t&			    operator=( self_t &&other ) = delete;
    /*!	@brief Evaluation operator.
     *
     *  Evaluate this RF function at the input point.
     *
     *	@param [in]			coords 	   coordinates of the input point.
    */
    coord_t 				operator()( const point_t &coords ) const
    {
      mFunct( norm2( coords - center )/radius );
    }

    // Getter(s)/Info --------------------------------------------- //
    public:
    /*!	@brief Returns the nr. of additional parameters of this RF.
     *
     *  By default it is assumed that the radial function
     *  does not depend on any additional parameter.
    */
    virtual std::size_t     getNumberOfParameters() const { return 0; };
    /*!	@brief Returns (true) if this RF has a compact support.
     *
     *  By default, it is assumed  that the RF has compact support.
     *
     *  @note If a family of radial basis function does not have a compact support,
     *  the derived class implementing the radial function, should override this method.
    */
    bool 			              hasCompactSupport() const { return mHasCompactSupport; };
    /*!	@brief Returns const pointer to the internal array of parameters
     *	of this RF.
     *
     *  The default behavior is to assume that the radial function does not
     *  depend on any additional parameter.
    */
    virtual const coord_t*	getParameters() const { return nullptr; };
    /*! @brief Returns the type of this radial function. */
    eRBFType                getType() const { return mType; }

    // Setter(s) -------------------------------------------------- //
    public:
    /*! @brief Set default values for function parameters, ie.:
     *  * radius = 1          (assumes uniform behavior across the basis, and normalized space)
     *  * center = (0, 0, 0)  (radial function centered in the origin)
    */
    virtual void            setDefault()
    {
      radius = (CoordT) 1;
      center.fill( (CoordT)0 );
    }
    /*!	@brief Set the value of the parameters of this Radial Function.
     *
     *  The default behavior is to assume that the radial function does not
     *  depend on any additional parameters.
    */
    virtual void            setParameters( const coord_t * ) {};

  }; //end class RF

  // ============================================================= //
  // DEFINITION OF CLASS RFP                                       //
  // ============================================================= //
  /*! @brief Definition of generalized RF.
   *
   *  @tparam       Dim           nr. of dimensions in the working space (Dim>0)
   *  @tparam       NParams       Nr of additional parameters for this RF (NParams>0)
   *  @tparam       CoordT        (default = double) type of coefficients, e.g. double, float, etc.
   *                              (only scalar floating point types are allowed).
   *
   *  This class can store a generic radial function which depends
   *  on a arbitrary nr. of parameters of type CoordT.
   *
   *  ** Usage **
   *  * Default usage *
   *  If you wish to use one of the radial function implemented in bitpit just call:
   *  RF::New (providing the specific type you want to use).
   *
   *  * Passing the ownership of the additional parameters to RFP class. *
   *  If you wish to encapsulate a user-defined function which depends on N parameters
   *  you can invoke the RFP constructor by a pointer to the function which implements
   *  the particular expression of your radial function,
   *  and specifying the nr. of additional parameters for this function.
   *  For instance, assumung D=3, CoordT = double and a user-defined funtion which depends
   *  on 2 additional parameters:
   *
   *  CoordT = my_radial_funct( CoordT r, CoordT par1, CoordT par2 ) defined somewhere
   *  auto my_rf = new RFP<3, 2, double>(
   *    bitpit::rbf::eRBFType::kUserDefined,
   *    &my_radial_funct
   *  )
   *
   *  In this case the instance my_rf will take exclusive ownership of the additional parameters
   *  (par1 and par2) and will bind such parameters to the function provided as input
   *  at construction time. The value of there parameters can be accesed/modified
   *  any time via #setParameters and #getParameters
   *
   *  In some cases the user might want to keep ownership of some (all) of the additional parameters.
   *  You can keep the ownership of such parameters by binding yourself the desired parameters
   *  and leaving RFP the task of taking the ownership of the remaining parameters.
   *  For instance, assuming D=3, CoordT = double, and a user-defined function
   *  whihc depends on 3 parameters:
   *
   *  CoordT = my_radial_funct_2( CoordT r, Coord_T par1, CoordT par2, CoordT par3 ) defined somewhere,
   *  double par1 = //somevalue,
   *         par3 = //somevalue
   *  auto my_rf = new RFP<3, 1 //nr. of parameters that will be managed by RFP// , double>(
   *    bitpit::rbf::eRBFType::kUserDefined,
   *    std::bind( &my_radial_funct_2, std::placeholders::_1, std::cref(par1), std::placeholders::_2, std::cred(par3)
   *  );
   *
   *  In this case, you will keep the ownership over 2 parameters (par1 and par3) while
   *  leaving the ownership of only 1 parameter (par2) to my_rf.
  */
  template<
    std::size_t Dim,
    std::size_t NParams,
    class CoordT = double
  >
  class RFP : public RF<Dim, CoordT>
  {
    // Static asserts ============================================ //
    static_assert(
      (NParams > 0),
      "bitpit::rbf::RFP<Dim, NParams, CoordT>: "
      "**ERROR** The nr. of additional parameters must be greater than 0. "
      "If the radial function does not depends on any parameter, use RF<Dim, CoordT>"
    );

    // Typedef(s) ================================================ //
    private:
    /*! @brief Type of the base class. */
    using base_t      = RF<Dim, CoordT>;
    /*! @brief Type of this class. */
    using self_t      = RFP<Dim, NParams, CoordT >;
    public:
    /*! @brief Type of functor holding the actual expression of the RF. */
    using rf_funct_t  = typename base_t::rf_funct_t;
    /*! @brief Coeff. type. */
    using coord_t     = typename base_t::coord_t;
    /*! @brief Point type. */
    using point_t     = typename base_t::point_t;

    // Friendships =============================================== //
    friend class RF<Dim, CoordT>;

    // Member variable(s) ======================================== //
    protected:
    using base_t::mType;
    using base_t::mFunct;
    /*! @brief List of values for the additional parameters of this radial function. */
    coord_t     mParams[NParams];
    public:
    using base_t::center;
    using base_t::radius;

    // Typedef(s) ================================================ //
    private:
    /*! @brief Default constructor.
     *
     *  Initialize a generalized multiquadric radial function with default values
     *  for the parameters.
    */
    RFP() :
      base_t()
    {}
    public:
    /*! @brief Contructor #1.
     *
     *  Initialize a generalized multiquadrics radial function with the specified
     *  values for the parameters.
     *
     *  @param [in]       r         radius of this function.
     *  @param [in]       c         center of this radial function.
     *  @param [in]       bias      bias coeff.
    */
    template<class ...Args>
    RFP( bitpit::rbf::eRBFType type, coord_t(*f)( coord_t, Args ...args) ) :
      base_t( type, bindParameters(f, mParams) )
    { }
    /*! @brief Copy constructor (deleted) */
    RFP( const self_t & ) = delete;
    /*! @brief Move-constructor (deleted). */
    RFP( self_t && ) = delete;

    // Operator(s) =============================================== //
    public:
    /*! @brief Copy-assignment operator (delete) */
    self_t&       operator=( const self_t & ) = delete;
    /*! @brief Move-assignement operator (delete) */
    self_t&       operator=( self_t &&other ) = delete;

    // Member function(s) ======================================== //

    // Getter(s)/Info -------------------------------------------- //
    public:

    /*! @brief Returns the nr. of additional parameters for this radial function. */
    virtual size_t          getNumberOfParameters() const override { return NParams; };
    /*! @brief Returns (const) pointer to the list of additional parameters of this funciton. */
    virtual const coord_t*  getParameters() const override { return mParams; }

    // Setter(s) ------------------------------------------------- //
    private:
    /*! @brief Bind the expression of the radial function to the parameters of
     *  this instance.
    */
    template<class f_in_t, std::size_t... I>
    static rf_funct_t       doBind( f_in_t f, coord_t* data, bitpit::index_sequence<I...> )
    {
         return std::bind( f, std::placeholders::_1, std::cref(data[I]...) ); // A trick here
    }
    template<class f_in_t>
    static rf_funct_t       bindParameters( f_in_t f, coord_t* data )
    {
         return doBind( f, data, bitpit::make_index_sequence<NParams>{} );
    }
    public:
    /*! @brief Set the value for the additional parameters of this function.*/
    virtual void            setParameters( const coord_t *values ) override
    {
      std::copy( values, values + NParams, mParams );
    }
  }; //end class GeneralizedMultiquadrics

  // ================================================================ //
  // EXPLICIT SPECIALIZATIONS                                         //
  // ================================================================ //
  extern template class RF<1, float>;
  extern template class RF<2, float>;
  extern template class RF<3, float>;
  extern template class RF<1, double>;
  extern template class RF<2, double>;
  extern template class RF<3, double>;
  extern template class RF<1, long double>;
  extern template class RF<2, long double>;
  extern template class RF<3, long double>;

  extern template class RFP<1, 1, float>;
  extern template class RFP<2, 1, float>;
  extern template class RFP<3, 1, float>;
  extern template class RFP<1, 1, double>;
  extern template class RFP<2, 1, double>;
  extern template class RFP<3, 1, double>;
  extern template class RFP<1, 1, long double>;
  extern template class RFP<2, 1, long double>;
  extern template class RFP<3, 1, long double>;

  extern template class RFP<1, 2, float>;
  extern template class RFP<2, 2, float>;
  extern template class RFP<3, 2, float>;
  extern template class RFP<1, 2, double>;
  extern template class RFP<2, 2, double>;
  extern template class RFP<3, 2, double>;
  extern template class RFP<1, 2, long double>;
  extern template class RFP<2, 2, long double>;
  extern template class RFP<3, 2, long double>;
  /*! @} */

} //end namespace rbf

} //end namespace bitpit
