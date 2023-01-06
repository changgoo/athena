#ifndef PARTICLES_PARTICLES_HPP_
#define PARTICLES_PARTICLES_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
//======================================================================================
//! \file particles.hpp
//! \brief defines classes for particle dynamics.
//======================================================================================

// C/C++ Standard Libraries
#include <string>
#include <vector>

// Athena headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../outputs/outputs.hpp"
#include "../parameter_input.hpp"
#include "particle_buffer.hpp"
#include "particle_gravity.hpp"
#include "particle_mesh.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// Forward definitions
class ParticleGravity;

//--------------------------------------------------------------------------------------
//! \struct Neighbor
//  \brief defines a structure for links to neighbors

struct Neighbor {
  NeighborBlock *pnb;
  MeshBlock *pmb;
  Neighbor *next, *prev;

  Neighbor() : pnb(NULL), pmb(NULL), next(NULL), prev(NULL) {}
};

//----------------------------------------------------------------------------------------
//! \struct ParticleParameters
//! \brief container for parameters read from `<particle?>` block in the input file

struct ParticleParameters {
  int block_number, ipar;
  bool table_output, gravity;
  std::string block_name;
  std::string partype;

  ParticleParameters() : block_number(0), ipar(-1), table_output(false), gravity(false) {}
};

//--------------------------------------------------------------------------------------
//! \class Particles
//! \brief defines the base class for all implementations of particles.

class Particles {
friend class ParticleGravity;
friend class ParticleMesh;

 public:
  // Class constant
  static const int NHISTORY = 8;  //!> number of variables in history output

  // Constructor
  Particles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  virtual ~Particles();

  // Static functions
  // TODO(SMOON) some of these could be changed to member function
  // TODO(SMOON) if they are helper functions, take them out of the
  //             class for better readerbility, since they are not
  //             a part of the interface
  // SMOON: maybe we need Mesh-level pure abstract interface class.
  static void AMRCoarseToFine(Particles *pparc, Particles *pparf, MeshBlock* pmbf);
  static void AMRFineToCoarse(Particles *pparc, Particles *pparf);
  // TODO(SMOON) bad function name?
  static void FindDensityOnMesh(Mesh *pm, bool include_momentum);
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void PostInitialize(Mesh *pm, ParameterInput *pin);
  static void FormattedTableOutput(Mesh *pm, OutputParameters op);
  static void GetHistoryOutputNames(std::string output_names[], int ipar);
  static std::int64_t GetTotalNumber(Mesh *pm);

  // Methods (interface)
  // TODO(SMOON) Potentially better approach might be not overriding AddOneParticle at all
  // In principle, the one who creates a particle will always know which particle they
  // want to create. Therefore, it may be more natural to <dynamic_cast> Particle to a
  // specific derived Particles and call the exact version; the idea is that, because
  // we know which Particle the base pointer points to when creating a particle,
  // the particle creation function does not have to be polymorphic.
  virtual void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3) {}
  virtual void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3, Real taus) {}
  void RemoveOneParticle(int k);
  void DepositPMtoMesh(int stage); // TODO(SMOON) should be moved to ParticleMesh
  // TODO(SMOON) (template method pattern is appropriate here)
  virtual void Integrate(int step);
  virtual Real NewBlockTimeStep();

  std::size_t GetSizeInBytes() const;
  bool IsGravity() const { return isgravity_; }
  int GetNumPar() const { return npar_; }
  // Read-only accessors to the particle properties
  const AthenaArray<int>& pid() const { return pid_; }
  const AthenaArray<Real>& mass() const { return mass_; }
  const AthenaArray<Real>& xp0() const { return xp0_; }
  const AthenaArray<Real>& yp0() const { return yp0_; }
  const AthenaArray<Real>& zp0() const { return zp0_; }
  const AthenaArray<Real>& vpx0() const { return vpx0_; }
  const AthenaArray<Real>& vpy0() const { return vpy0_; }
  const AthenaArray<Real>& vpz0() const { return vpz0_; }

  // ************************** //
  // Input/Output API //
  // ************************** //
  void UnpackParticlesForRestart(char *mbdata, std::size_t &os);
  void PackParticlesForRestart(char *&pdata);
  void AddHistoryOutput(Real data_sum[], int pos);
  void OutputParticles(bool header); // individual particle history;
  void OutputParticles(bool header, int kid);
  // TODO(SMOON) may not be needed; why not just make it default?
  void ToggleParHstOutFlag();

  // ************************** //
  // Boundary communication API //
  // ************************** //
  void ClearBoundary();
  void ClearNeighbors();
  void LinkNeighbors(MeshBlockTree &tree, int64_t nrbx1, int64_t nrbx2, int64_t nrbx3,
                     int root_level);
  void LoadParticleBuffer(ParticleBuffer *ppb, int k);
#ifdef MPI_PARALLEL
  void SendParticleBuffer(ParticleBuffer& send, int dst);
  void ReceiveParticleBuffer(int nb_rank, ParticleBuffer& recv,
                             enum BoundaryStatus& bstatus);
#endif
  void SendToNeighbors();
  void SetPositionIndices(); // TODO(SMOON) Flatten this function and move to protected
  bool ReceiveFromNeighbors();
  void StartReceivingParticlesShear();
  void SendParticlesShear();
  int FindTargetGidAlongX2(Real x2);
  void ClearBoundaryShear();
  bool ReceiveFromNeighborsShear();

  // Data members
  // number of particle containers
  static int num_particles, num_particles_grav, num_particles_output;
  ParticleMesh *ppm;  //!> ptr to particle-mesh
  const int ipar;     // index of this Particle in ppars vector
  std::string input_block_name, partype; // TODO(SMOON) input_block_name bad design?

 protected:
  // Protected interfaces (to be used by derived classes)
  // TODO(SMOON) avoid call super using template method
  virtual void AssignShorthands();  //!> Needs to be called everytime
                                    //!> intprop, realprop, & auxprop are resized
                                    //!> Be sure to call back when derived.
  virtual void AllocateMemory();    //!> Needs to be called in the derived class init
  int AddIntProperty();
  int AddRealProperty();
  int AddAuxProperty();
  int AddWorkingArray();
  void UpdateCapacity(int new_nparmax);  //!> Change the capacity of particle arrays
  bool CheckInMeshBlock(Real x1, Real x2, Real x3);
  void SaveStatus(); // x->x0, v->v0

  // Data members
  // Shallow slices of the actual data container (intprop, realprop, auxprop, work)
  AthenaArray<int> pid_;                  //!> particle ID
  AthenaArray<Real> mass_;                //!> mass
  AthenaArray<Real> xp_, yp_, zp_;        //!> position
  AthenaArray<Real> vpx_, vpy_, vpz_;     //!> velocity
  AthenaArray<Real> xi1_, xi2_, xi3_;     //!> position indices in local meshblock
  AthenaArray<Real> xp0_, yp0_, zp0_;     //!> beginning position (SMOON: What is this?)
  AthenaArray<Real> vpx0_, vpy0_, vpz0_;  //!> beginning velocity (SMOON: What is this?)

  int npar_;     //!> number of particles
  int nparmax_;  //!> maximum number of particles per meshblock
  Real cfl_par_;  //!> CFL number for particles
  std::vector<std::string> intfieldname, realfieldname, auxfieldname;

  ParticleGravity *ppgrav; //!> ptr to particle-gravity
  MeshBlock* pmy_block;  //!> MeshBlock pointer
  Mesh* pmy_mesh_;        //!> Mesh pointer

  // shearing box parameters
  Real Omega_0_, qshear_, qomL_;
  int ShBoxCoord_;
  bool orbital_advection_defined_;

  // The actual data storage of all particle properties
  // Note to developers:
  // Direct access to these containers is discouraged; use shorthands instead.
  // e.g.) use mass_(k) instead of realprop(imass, k)
  // Auxiliary properties (auxprop) is communicated when particles moving to
  // another meshblock. Working arrays (work) is not communicated.
  AthenaArray<int> intprop;
  AthenaArray<Real> realprop, auxprop, work;

 private:
  // Class method
  static void ProcessNewParticles(Mesh *pmesh, int ipar);

  // Methods (implementation)
  // Need to be implemented in derived classes
  // TODO(SMOON) Functions such as SourceTerms, EulerStep, BorisKick, ... needs to be
  // inside the implementation of template function Integrate(). This needs some
  // consistent name convention.
  // for example, the interface looks like
  // Integrate () {
  //   Kick()
  //   Drift()
  //   Kick()
  // }
  // Where the actual implementation of Kick would be either EulerStep or BorisKick, etc.
  virtual void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                             AthenaArray<Real>& meshdst)=0;

  int CountNewParticles() const;
  void ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3);
  void EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc);
  void FlushReceiveBuffer(ParticleBuffer& recv);
  void GetPositionIndices(int npar,
                          const AthenaArray<Real>& xp,
                          const AthenaArray<Real>& yp,
                          const AthenaArray<Real>& zp,
                          AthenaArray<Real>& xi1,
                          AthenaArray<Real>& xi2,
                          AthenaArray<Real>& xi3);
  void SetNewParticleID(int id);
  struct Neighbor* FindTargetNeighbor(
      int ox1, int ox2, int ox3, int xi1, int xi2, int xi3);
  void ApplyBoundaryConditionsShear(int k, Real &x1, Real &x2, Real &x3);
  void OutputOneParticle(std::ostream &os, int k, bool header);

  // Class variable
  static std::vector<int> idmax;
  static bool initialized;  //!> whether or not the class is initialized
  static ParameterInput *pinput;

  // Data members

  int nint;          //!> numbers of integer particle properties
  int nreal;         //!> numbers of real particle properties
  int naux;          //!> number of auxiliary particle properties
  int nwork;         //!> number of working arrays for particles
  int nint_buf, nreal_buf; //!> number of properties for buffer

  int ipid;                 //!> index for the particle ID
  int imass;                // index for the particle mass
  int ixp, iyp, izp;        // indices for the position components
  int ivpx, ivpy, ivpz;     // indices for the velocity components

  int ixp0, iyp0, izp0;     // indices for beginning position components
  int ivpx0, ivpy0, ivpz0;  // indices for beginning velocity components

  int ixi1, ixi2, ixi3;     // indices for position indices

  int igx, igy, igz; // indices for gravity force

  int ish;

  bool parhstout_; //!> flag for individual particle history output
  bool isgravity_; //!> flag for gravity
  bool active1_, active2_, active3_;  // active dimensions

  // MeshBlock-to-MeshBlock communication:
  BoundaryValues *pbval_;                            //!> ptr to my BoundaryValues
  Neighbor neighbor_[3][3][3];                       //!> links to neighbors
  ParticleBuffer recv_[56], recv_sh_[8];             //!> particle receive buffers
  enum BoundaryStatus bstatus_[56], bstatus_recv_sh_[8];  //!> boundary status
#ifdef MPI_PARALLEL
  static MPI_Comm my_comm;   //!> my MPI communicator
  ParticleBuffer send_[56],send_sh_[8];  //!> particle send buffers
  enum BoundaryStatus bstatus_send_sh_[8];  //!> comm. flags
#endif
};

//--------------------------------------------------------------------------------------
//! \class DustParticles
//! \brief defines the class for dust particles that interact with the gas via drag
//!        force.

class DustParticles : public Particles {

 public:
  // Constructor
  DustParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~DustParticles();

  // Methods (interface)
  void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3) override;
  void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3, Real taus) override;
  bool GetBackReaction() const { return backreaction; }
  bool GetDragForce() const { return dragforce; }
  bool IsVariableTaus() const { return variable_taus; }
  Real GetStoppingTime() const { return taus0; }
  Real NewBlockTimeStep() override;

 private:
  // Methods (implementation)
  void AssignShorthands() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                     AthenaArray<Real>& meshdst) override;
  void UserStoppingTime(Real t, Real dt, const AthenaArray<Real>& meshsrc);

  // Data members
  bool backreaction;   //!> turn on/off back reaction
  bool dragforce;      //!> turn on/off drag force
  bool variable_taus;  //!> whether or not the stopping time is variable

  int iwx, iwy, iwz;         // indices for working arrays
  int itaus;                 //!> index for stopping time

  Real taus0;  //!> constant/default stopping time (in code units)
  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays
  AthenaArray<Real> taus_;              // shorthand for stopping time
};

//--------------------------------------------------------------------------------------
//! \class TracerParticles
//! \brief defines the class for velocity Tracer particles

class TracerParticles : public Particles {

 public:
  // Constructor
  TracerParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~TracerParticles();

  // Methods (interface)
  void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3) override;

 private:
  // Methods (implementation)
  void AssignShorthands() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                     AthenaArray<Real>& meshdst) override;

  // Data members
  int iwx, iwy, iwz;         // indices for working arrays
  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays
};

//--------------------------------------------------------------------------------------
//! \class StarParticles
//! \brief defines the class for Star particles

class StarParticles : public Particles {

 public:
  // Constructor
  StarParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~StarParticles();

  // Methods (interface)
  void Integrate(int step) override;
  void AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3) override;

 private:
  // Methods (implementation)
  void AssignShorthands() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                     AthenaArray<Real>& meshdst) override;

  void Kick(Real t, Real dt, const AthenaArray<Real>& meshsrc);
  void Drift(Real t, Real dt);
  void BorisKick(Real t, Real dt);
  void Age(Real t, Real dt);

  void ExertTidalForce(Real t, Real dt);
  void PointMass(Real t, Real dt, Real gm);
  void ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3);

  // Data members
  Real dt_old;
  // TODO(SMOON) index and variable name are inconsistent
  int imetal, iage; // indices for additional Real properties
  int igas;                // indices for additional Aux properties
  AthenaArray<Real> mzp, tage;        // shorthand for real properties
  AthenaArray<Real> fgas;                     // shorthand for aux properties
};

#endif  // PARTICLES_PARTICLES_HPP_
