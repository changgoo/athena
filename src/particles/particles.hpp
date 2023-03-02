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
#include <limits>
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
  int block_number, ipar, nghost;
  bool table_output, gravity;
  std::string block_name;
  std::string partype;
  // TODO(SMOON) Add nhistory variable
  ParticleParameters() : block_number(0), ipar(-1), nghost(0), table_output(false),
                         gravity(false) {}
};

//--------------------------------------------------------------------------------------
//! \class Particles
//! \brief defines the base class for all implementations of particles.

class Particles {
friend class ParticleGravity;
friend class ParticleMesh;

 public:
  // TODO(SMOON) this must be variable, initialized through ParticleParameters,
  // because different particle species may have different number of history output.
  static const int NHISTORY = 8;  //!> number of variables in history output

  // Constructor
  Particles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  virtual ~Particles();

  // Particle interface
  int AddOneParticle(Real mp, Real x1, Real x2, Real x3, Real v1, Real v2, Real v3);
  void RemoveOneParticle(int k);
  virtual void Integrate(int step); // TODO(SMOON) apply template pattern?
  Real NewBlockTimeStep();
  std::size_t GetSizeInBytes() const;
  bool IsGravity() const { return isgravity_; }
  int GetNumPar() const { return npar_; }
  void GridIndex(Real xp, Real yp, Real zp, int &ip, int &jp, int &kp) const;
  virtual void InteractWithMesh() {};

  // Input/Output interface
  void UnpackParticlesForRestart(char *mbdata, std::size_t &os);
  void PackParticlesForRestart(char *&pdata);
  void AddHistoryOutput(Real data_sum[], int pos);
  void OutputParticles(bool header); // individual particle history;
  void OutputParticles(bool header, int kid);
  void ToggleParHstOutFlag();

  // Boundary communication interface (defined in particles_bvals.cpp)
  void ClearBoundary();
  void ClearNeighbors();
  void LinkNeighbors(MeshBlockTree &tree, int64_t nrbx1, int64_t nrbx2, int64_t nrbx3,
                     int root_level);
  void LoadParticleBuffer(ParticleBuffer *ppb, int k, bool ghost=false);
#ifdef MPI_PARALLEL
  void SendParticleBuffer(ParticleBuffer& send, int dst);
  void ReceiveParticleBuffer(int nb_rank, ParticleBuffer& recv,
                             enum BoundaryStatus& bstatus);
#endif
  void SendToNeighbors();
  bool ReceiveFromNeighbors();
  void StartReceivingParticlesShear();
  void SendParticlesShear();
  int FindTargetGidAlongX2(Real x2);
  void ClearBoundaryShear();
  bool ReceiveFromNeighborsShear();

  // Static functions
  static void AMRCoarseToFine(Particles *pparc, Particles *pparf, MeshBlock* pmbf);
  static void AMRFineToCoarse(Particles *pparc, Particles *pparf);
  static void ComputePMDensityAndCommunicate(Mesh *pm, bool include_momentum);
  static void Initialize(Mesh *pm, ParameterInput *pin);
  static void PostInitialize(Mesh *pm, ParameterInput *pin);
  static void FormattedTableOutput(Mesh *pm, OutputParameters op);
  static void GetHistoryOutputNames(std::string output_names[], int ipar);
  static std::int64_t GetTotalNumber(Mesh *pm);

  // Data members
  // number of particle containers
  static int num_particles, num_particles_grav, num_particles_output;
  ParticleMesh *ppm;  //!> ptr to particle-mesh
  const int ipar;     // index of this Particle in ppars vector
  std::string input_block_name, partype;

  // Shallow slices of the actual data container (intprop, realprop, auxprop, work)
  AthenaArray<int> pid, sh;                  //!> particle ID
  AthenaArray<Real> mass;                //!> mass
  AthenaArray<Real> xp, yp, zp;        //!> position
  AthenaArray<Real> vpx, vpy, vpz;     //!> velocity
  AthenaArray<Real> xp0, yp0, zp0;     //!> beginning position (SMOON: What is this?)
  AthenaArray<Real> vpx0, vpy0, vpz0;  //!> beginning velocity (SMOON: What is this?)

 protected:
  // Protected interfaces (to be used by derived classes)
  // SMOON: A possibility of forgetting to call these two functions may indicate
  // an antipattern.
  void AssignShorthands(); //!> Needs to be called in the derived class constructor
  void AllocateMemory();   //!> Needs to be called in the derived class constructor
  int AddIntProperty();
  int AddRealProperty();
  int AddAuxProperty();
  int AddWorkingArray();
  void UpdateCapacity(int new_nparmax);  //!> Change the capacity of particle arrays
  bool CheckInMeshBlock(Real x1, Real x2, Real x3);
  // TODO(SMOON) these two functions  can be moved to private once Integrate follows
  // template design pattern.
  void SaveStatus(); // x->x0, v->v0
  void UpdatePositionIndices();

  int npar_;     //!> number of particles
  int npar_gh_;     //!> number of ghost particles
  int nparmax_;  //!> maximum number of particles per meshblock
  const int nghost_; //!> number of ghost cells within which ghost particles are communicated
  Real cfl_par_;  //!> CFL number for particles

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
  // e.g.) use mass(k) instead of realprop(imass, k)
  // Auxiliary properties (auxprop) is communicated when particles moving to
  // another meshblock. Working arrays (work) is not communicated.
  std::vector<std::string> intpropname, realpropname, auxpropname;
  AthenaArray<int> intprop;
  AthenaArray<Real> realprop, auxprop, work;

 private:
  // Methods (implementation)
  // Need to be implemented in derived classes
  virtual void AssignShorthandsForDerived()=0;
  virtual void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc)=0;
  virtual void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                             AthenaArray<Real>& meshdst) {}
  void UpdatePositionIndices(int npar,
                             const AthenaArray<Real>& xp,
                             const AthenaArray<Real>& yp,
                             const AthenaArray<Real>& zp,
                             AthenaArray<Real>& xi1,
                             AthenaArray<Real>& xi2,
                             AthenaArray<Real>& xi3);
  int CountNewParticles() const;
  void SetNewParticleID(int id);
  // hooks for further timestep constraints for derived particles
  virtual Real NewDtForDerived() { return std::numeric_limits<Real>::max(); }
  void EulerStep(Real t, Real dt, const AthenaArray<Real>& meshsrc);

  // Input/Output
  void OutputOneParticle(std::ostream &os, int k, bool header);

  // boundary conditions (implemented in particles_bvals.cpp)
  void ApplyBoundaryConditions(int k, Real &x1, Real &x2, Real &x3, bool ghost=false);
  void FlushReceiveBuffer(ParticleBuffer& recv, bool ghost=false);
  struct Neighbor* FindTargetNeighbor(
      int ox1, int ox2, int ox3, int xi1, int xi2, int xi3);
  void ApplyBoundaryConditionsShear(int k, Real &x1, Real &x2, Real &x3);

  // Private static function (called in static interfaces)
  static void ProcessNewParticles(Mesh *pmesh, int ipar);

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

  // indices for intprop shorthands
  int ipid;                 // index for the particle ID
  int ish;                  // index for shear boundary flag

  // indices for realprop shorthands
  int imass;                // index for the particle mass
  int ixp, iyp, izp;        // indices for the position components
  int ivpx, ivpy, ivpz;     // indices for the velocity components

  // indices for auxprop shorthands
  int ixp0, iyp0, izp0;     // indices for beginning position components
  int ivpx0, ivpy0, ivpz0;  // indices for beginning velocity components

  // indices for work shorthands
  int ixi1, ixi2, ixi3;     // indices for position indices
  int igx, igy, igz; // indices for gravity force

  AthenaArray<Real> xi1_, xi2_, xi3_;     //!> position indices in local meshblock

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
  ParticleBuffer send_[56], send_gh_[56], send_sh_[8];  //!> particle send buffers
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
  bool GetBackReaction() const { return backreaction; }
  bool GetDragForce() const { return dragforce; }
  bool IsVariableTaus() const { return variable_taus; }
  Real GetStoppingTime() const { return taus0; }

  // Data members
  // shorthand for additional properties
  AthenaArray<Real> taus;              // shorthand for stopping time

 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void DepositToMesh(Real t, Real dt, const AthenaArray<Real>& meshsrc,
                     AthenaArray<Real>& meshdst) override;
  void UserStoppingTime(Real t, Real dt, const AthenaArray<Real>& meshsrc);
  Real NewDtForDerived() override;

  // Data members
  bool backreaction;   //!> turn on/off back reaction
  bool dragforce;      //!> turn on/off drag force
  bool variable_taus;  //!> whether or not the stopping time is variable

  // indicies for additional shorthands
  int itaus;                 // index for stopping time
  int iwx, iwy, iwz;         // indices for working arrays

  AthenaArray<Real> wx, wy, wz;        // shorthand for working arrays

  Real taus0;  //!> constant/default stopping time (in code units)
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

  // Data members
  // shorthand for additional properties


 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;

  // Data members
  // indicies for additional shorthands
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
  // TODO(SMOON) should be removed after applying template design pattern
  void Integrate(int step) override;

  // Data members
  // shorthand for additional properties
  AthenaArray<Real> metal, age, fgas;

 private:
  // Methods (implementation)
  void AssignShorthandsForDerived() override;
  void SourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void UserSourceTerms(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;
  void ReactToMeshAux(Real t, Real dt, const AthenaArray<Real>& meshsrc) override;

  void Kick(Real t, Real dt, const AthenaArray<Real>& meshsrc);
  void Drift(Real t, Real dt);
  void BorisKick(Real t, Real dt);
  void Age(Real t, Real dt);

  void ExertTidalForce(Real t, Real dt);
  void PointMass(Real t, Real dt, Real gm);
  void ConstantAcceleration(Real t, Real dt, Real g1, Real g2, Real g3);

  // Data members
  Real dt_old;
  // indicies for additional shorthands
  int imetal, iage, ifgas;            // indices for:
};

//--------------------------------------------------------------------------------------
//! \class SinkParticles
//! \brief defines the class for Sink particles

class SinkParticles : public StarParticles {
 public:
  // Constructor
  SinkParticles(MeshBlock *pmb, ParameterInput *pin, ParticleParameters *pp);

  // Destructor
  ~SinkParticles();

  // Methods (interface)
  void InteractWithMesh() override;
  void SetControlVolume();

  // Data members
  const int rctrl = 1; // Extent of the control volume. The side length of the control
                       // volume is 2*rctrl + 1.

 private:
  // Methods (implementation)
  void AccreteMass();
  void SetControlVolume(AthenaArray<Real> &cons, int ip, int jp, int kp);

  // Data members
};

#endif  // PARTICLES_PARTICLES_HPP_
