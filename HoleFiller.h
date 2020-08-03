#pragma once


#pragma warning(push)
#pragma warning(disable:4244)
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyConnectivity.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/Traits.hh>
#include <OpenMesh/Tools/Smoother/JacobiLaplaceSmootherT.hh>
#pragma warning(pop)

enum class FixType
{
	idMinArea, //针对小洞
	idMinAreaMaxDiheral, //针对大洞
	idMinAreaMaxDiheralNormal, //Todo
};

class HoleFiller
{
	using Triangle_mesh = OpenMesh::TriMesh_ArrayKernelT<>;
	using Edge = Triangle_mesh::Edge;
	using Halfedge = Triangle_mesh::Halfedge;
	using FaceHandle = Triangle_mesh::FaceHandle;
	using VertexHandle = Triangle_mesh::VertexHandle;
	using EdgeHandle = Triangle_mesh::EdgeHandle;
	using HalfedgeHandle = Triangle_mesh::HalfedgeHandle;
public: //预处理
	static void SmoothMeshBoundary(Triangle_mesh& mesh);
public:
	static bool hole_fillC2(Triangle_mesh& mesh, HalfedgeHandle& hh, FixType type = FixType::idMinAreaMaxDiheral);
	static bool hole_fillC1(Triangle_mesh& mesh, HalfedgeHandle& hh, FixType type = FixType::idMinAreaMaxDiheral);
	static bool hole_fillC0(Triangle_mesh& mesh, HalfedgeHandle& hh, FixType type = FixType::idMinAreaMaxDiheral);

};
