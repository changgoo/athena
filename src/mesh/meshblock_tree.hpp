#ifndef MESH_MESHBLOCK_TREE_HPP_
#define MESH_MESHBLOCK_TREE_HPP_
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file meshblock_tree.hpp
//  \brief defines the LogicalLocation structure and MeshBlockTree class
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../defs.hpp"

class Mesh;

//--------------------------------------------------------------------------------------
//! \class MeshBlockTree
//  \brief Objects are nodes in an AMR MeshBlock tree structure

class MeshBlockTree {
  friend class Mesh;
  friend class MeshBlock;
  friend class BoundaryBase;
 public:
  explicit MeshBlockTree(Mesh *pmesh);
  MeshBlockTree(MeshBlockTree *parent, int ox1, int ox2, int ox3);
  ~MeshBlockTree();

  // accessor
  MeshBlockTree* GetLeaf(int ox1, int ox2, int ox3)
  { return pleaf_[(ox1 + (ox2<<1) + (ox3<<2))]; }
  int GetGID() {return gid_;}

  // functions
  void CreateRootGrid();
  void AddMeshBlock(LogicalLocation rloc, int &nnew);
  void AddMeshBlockWithoutRefine(LogicalLocation rloc);
  void Refine(int &nnew);
  void Derefine(int &ndel);
  MeshBlockTree* FindMeshBlock(LogicalLocation tloc);
  void CountMeshBlock(int& count);
  void GetMeshBlockList(LogicalLocation *list, int *pglist, int& count);
  MeshBlockTree* FindNeighbor(LogicalLocation myloc, int ox1, int ox2, int ox3,
                              bool amrflag=false);

 private:
  // data
  MeshBlockTree** pleaf_;
  int gid_;
  LogicalLocation loc_;

  static MeshBlockTree* proot_;
  static int nleaf_;
};

#endif // MESH_MESHBLOCK_TREE_HPP_
