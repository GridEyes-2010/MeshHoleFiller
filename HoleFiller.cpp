/*
				   GNU LESSER GENERAL PUBLIC LICENSE
					   Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.


  This version of the GNU Lesser General Public License incorporates
the terms and conditions of version 3 of the GNU General Public
License, supplemented by the additional permissions listed below.

  0. Additional Definitions.

  As used herein, "this License" refers to version 3 of the GNU Lesser
General Public License, and the "GNU GPL" refers to version 3 of the GNU
General Public License.

  "The Library" refers to a covered work governed by this License,
other than an Application or a Combined Work as defined below.

  An "Application" is any work that makes use of an interface provided
by the Library, but which is not otherwise based on the Library.
Defining a subclass of a class defined by the Library is deemed a mode
of using an interface provided by the Library.

  A "Combined Work" is a work produced by combining or linking an
Application with the Library.  The particular version of the Library
with which the Combined Work was made is also called the "Linked
Version".

  The "Minimal Corresponding Source" for a Combined Work means the
Corresponding Source for the Combined Work, excluding any source code
for portions of the Combined Work that, considered in isolation, are
based on the Application, and not on the Linked Version.

  The "Corresponding Application Code" for a Combined Work means the
object code and/or source code for the Application, including any data
and utility programs needed for reproducing the Combined Work from the
Application, but excluding the System Libraries of the Combined Work.

  1. Exception to Section 3 of the GNU GPL.

  You may convey a covered work under sections 3 and 4 of this License
without being bound by section 3 of the GNU GPL.

  2. Conveying Modified Versions.

  If you modify a copy of the Library, and, in your modifications, a
facility refers to a function or data to be supplied by an Application
that uses the facility (other than as an argument passed when the
facility is invoked), then you may convey a copy of the modified
version:

   a) under this License, provided that you make a good faith effort to
   ensure that, in the event an Application does not supply the
   function or data, the facility still operates, and performs
   whatever part of its purpose remains meaningful, or

   b) under the GNU GPL, with none of the additional permissions of
   this License applicable to that copy.

  3. Object Code Incorporating Material from Library Header Files.

  The object code form of an Application may incorporate material from
a header file that is part of the Library.  You may convey such object
code under terms of your choice, provided that, if the incorporated
material is not limited to numerical parameters, data structure
layouts and accessors, or small macros, inline functions and templates
(ten or fewer lines in length), you do both of the following:

   a) Give prominent notice with each copy of the object code that the
   Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the object code with a copy of the GNU GPL and this license
   document.

  4. Combined Works.

  You may convey a Combined Work under terms of your choice that,
taken together, effectively do not restrict modification of the
portions of the Library contained in the Combined Work and reverse
engineering for debugging such modifications, if you also do each of
the following:

   a) Give prominent notice with each copy of the Combined Work that
   the Library is used in it and that the Library and its use are
   covered by this License.

   b) Accompany the Combined Work with a copy of the GNU GPL and this license
   document.

   c) For a Combined Work that displays copyright notices during
   execution, include the copyright notice for the Library among
   these notices, as well as a reference directing the user to the
   copies of the GNU GPL and this license document.

   d) Do one of the following:

	   0) Convey the Minimal Corresponding Source under the terms of this
	   License, and the Corresponding Application Code in a form
	   suitable for, and under terms that permit, the user to
	   recombine or relink the Application with a modified version of
	   the Linked Version to produce a modified Combined Work, in the
	   manner specified by section 6 of the GNU GPL for conveying
	   Corresponding Source.

	   1) Use a suitable shared library mechanism for linking with the
	   Library.  A suitable mechanism is one that (a) uses at run time
	   a copy of the Library already present on the user's computer
	   system, and (b) will operate properly with a modified version
	   of the Library that is interface-compatible with the Linked
	   Version.

   e) Provide Installation Information, but only if you would otherwise
   be required to provide such information under section 6 of the
   GNU GPL, and only to the extent that such information is
   necessary to install and execute a modified version of the
   Combined Work produced by recombining or relinking the
   Application with a modified version of the Linked Version. (If
   you use option 4d0, the Installation Information must accompany
   the Minimal Corresponding Source and Corresponding Application
   Code. If you use option 4d1, you must provide the Installation
   Information in the manner specified by section 6 of the GNU GPL
   for conveying Corresponding Source.)

  5. Combined Libraries.

  You may place library facilities that are a work based on the
Library side by side in a single library together with other library
facilities that are not Applications and are not covered by this
License, and convey such a combined library under terms of your
choice, if you do both of the following:

   a) Accompany the combined library with a copy of the same work based
   on the Library, uncombined with any other library facilities,
   conveyed under the terms of this License.

   b) Give prominent notice with the combined library that part of it
   is a work based on the Library, and explaining where to find the
   accompanying uncombined form of the same work.

  6. Revised Versions of the GNU Lesser General Public License.

  The Free Software Foundation may publish revised and/or new versions
of the GNU Lesser General Public License from time to time. Such new
versions will be similar in spirit to the present version, but may
differ in detail to address new problems or concerns.

  Each version is given a distinguishing version number. If the
Library as you received it specifies that a certain numbered version
of the GNU Lesser General Public License "or any later version"
applies to it, you have the option of following the terms and
conditions either of that published version or of any later version
published by the Free Software Foundation. If the Library as you
received it does not specify a version number of the GNU Lesser
General Public License, you may choose any version of the GNU Lesser
General Public License ever published by the Free Software Foundation.

  If the Library as you received it specifies that a proxy can decide
whether future versions of the GNU Lesser General Public License shall
apply, that proxy's public statement of acceptance of any version is
permanent authorization for you to choose that version for the
Library.
 */

#include <Eigen/Eigen>
#include "HoleFiller.h"
#include <map>
#include <set>
#pragma warning(push)
#pragma  warning(disable : 4244 )
#pragma  warning(disable : 4267 )
#define INV_M_PI 0.31830988618379067153776752674503
#define INV_1_3 0.33333333333333333333333333333333
namespace {
	using Scalar = double;
	using Triangle_mesh = OpenMesh::TriMesh_ArrayKernelT<>;
	using Edge = Triangle_mesh::Edge;
	using Halfedge = Triangle_mesh::Halfedge;
	using FaceHandle = Triangle_mesh::FaceHandle;
	using VertexHandle = Triangle_mesh::VertexHandle;
	using EdgeHandle = Triangle_mesh::EdgeHandle;
	using HalfedgeHandle = Triangle_mesh::HalfedgeHandle;
	using Real = Triangle_mesh::Scalar;
	using Point = Triangle_mesh::Point;
	template<class Real>
	class NMatrix :public std::vector<Real>
	{
	protected:
		using BaseClass = std::vector<Real>;
	public:
		NMatrix() :BaseClass() {}
		NMatrix(int row, int col, Real val = Real()) :
			BaseClass(row*col, val), m_row(row), m_col(col) {}
		NMatrix(int row, int col, std::initializer_list<Real>& values)
			:BaseClass(values), m_row(row), m_col(col) {
			assert(values.size() == m_row * m_col&&"dim must be equal!");
		}
		NMatrix(int row, int col, std::vector<Real>& data) :
			BaseClass(data), m_row(row), m_col(col) {
			assert(data.size() == m_row * m_col&&"dim must be equal!");
		}
	public:
		const Real& operator()(int r, int c)const
		{
			return this->data()[r*m_col + c];
		}
		Real& operator()(int r, int c)
		{
			return this->data()[r*m_col + c];
		}
	public:
		int rows()const { return m_row; }
		int cols()const { return m_col; }
	public:
		void resize(int row, int col, Real v = Real()) {
			m_row = row; m_col = col;
			BaseClass::resize(row*col, v);
		}
	public:
		static int id(int r, int c, int col) {
			return r * col + c;
		}
	protected:
		int m_row, m_col;
	};


	Real distance(const Point& a, const  Point& b)
	{
		return (a - b).length();
	}
	Real angle(const Point& a, const Point& b)
	{
		Real x = dot(a, b) / (a.length()* b.length());
		return std::acos(x);
	}
	namespace Fair {
		inline static  float eps__(float) { return FLT_EPSILON; }
		inline static double eps__(double) { return DBL_EPSILON; }
		template <class T, typename Real>
		inline   bool IsZero(T a, Real eps) { return fabs(a) < eps; }
		template <class T>
		inline bool IsZero(T a) { return IsZero(a, eps__(a)); }
		template <typename T>
		inline bool IsEqual(T a, T b) { return IsZero(a - b); }

		template <typename T>
		inline  T TrigCut(T value)
		{
			if (value < -1)
			{
				value = -1;
			}
			else if (value > 1)
			{
				value = 1;
			}
			return value;
		}
		template <typename PointT>
		int IsObtuse(const PointT& p0, const PointT& p1, const PointT& p2)
		{
			const double a0 = ((p1 - p0) | (p2 - p0));
			if (a0 < 0.0)
				return 1;
			else
			{
				const double a1 = ((p0 - p1) | (p2 - p1));
				if (a1 < 0.0)
					return 2;
				else
				{
					const double a2 = ((p0 - p2) | (p1 - p2));
					if (a2 < 0.0)
						return 3;
					else return 0;
				}
			}
		}

		template < typename PointT, typename InputIterator >
		PointT Centroid(InputIterator begin, InputIterator end)
		{
			PointT point(0.0f, 0.0f, 0.0f);
			if (begin == end)
				return point;
			int npts = 0;
			while (begin != end)
			{
				point = point + *begin++;
				npts++;
			}
			//
			return point / (float)npts;
		}

		class SmoothMesh
		{
		public:
			static bool CotLaplaceMatrix(
				const Triangle_mesh& mesh,
				Eigen::SparseMatrix<Scalar>& L,
				Eigen::SparseMatrix<Scalar>& M)
			{
				using Point = OpenMesh::Vec3f;
				int vsize = static_cast<int>(mesh.n_vertices());
				if (vsize < 3)
					return false;
				if (!mesh.is_trimesh())
					return false;
				L.resize(vsize, vsize);
				M.resize(vsize, vsize);

				std::vector<Eigen::Triplet<Scalar, int> > IJV;
				std::vector<Eigen::Triplet<Scalar, int> > MTrip;
				IJV.reserve(8 * mesh.n_vertices());
				MTrip.reserve(mesh.n_vertices());
				auto v_end = mesh.vertices_end();
				for (auto v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
				{
					double w_ii = 0;
					double area = 0.0f;
					auto voh_it = mesh.cvoh_iter(*v_it);
					if (!voh_it->is_valid())
						return false;
					for (; voh_it.is_valid(); ++voh_it)
					{
						if (mesh.is_boundary(mesh.edge_handle(*voh_it)))
							continue;
						HalfedgeHandle lefthh = mesh.prev_halfedge_handle(*voh_it);
						HalfedgeHandle righthh = mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(*voh_it));
						const Point& p0 = mesh.point(*v_it);
						const Point& p1 = mesh.point(mesh.to_vertex_handle(*voh_it));
						const Point& p2 = mesh.point(mesh.from_vertex_handle(lefthh));
						const Point& p3 = mesh.point(mesh.to_vertex_handle(righthh));
						double alpha = std::acos(TrigCut((p0 - p2).normalize() | (p1 - p2).normalize()));
						double beta = std::acos(TrigCut((p0 - p3).normalize() | (p1 - p3).normalize()));
						//
						double cotw = 0.0;
						if (!IsEqual(alpha, M_PI / 2.0))
							cotw += 1.0 / std::tan(alpha);
						if (!IsEqual(beta, M_PI / 2.0))
							cotw += 1.0 / std::tan(beta);
						if (IsZero(cotw) || std::isinf(cotw) || std::isnan(cotw))
							continue;
						// calculate area
						const int obt = IsObtuse(p0, p1, p2);
						if (obt == 0)
						{
							double gamma = acos(TrigCut((p0 - p1).normalize() | (p2 - p1).normalize()));

							double tmp = 0.0;
							if (!IsEqual(alpha, M_PI / 2.0))
								tmp += (p0 - p1).sqrnorm()*1.0 / tan(alpha);

							if (!IsEqual(gamma, M_PI / 2.0))
								tmp += (p0 - p2).sqrnorm()*1.0 / tan(gamma);
							if (IsZero(tmp) || std::isinf(tmp) || std::isnan(tmp))
								continue;
							area += 0.125f*(tmp);
						}
						else
						{
							if (obt == 1)
							{
								area += ((p1 - p0) % (p2 - p0)).norm() * 0.5f * 0.5f;
							}
							else
							{
								area += ((p1 - p0) % (p2 - p0)).norm() * 0.5f * 0.25f;
							}
						}
						//
						w_ii -= cotw;
						VertexHandle vh = mesh.to_vertex_handle(*voh_it);
						IJV.push_back(Eigen::Triplet<Scalar, int>(v_it->idx(), vh.idx(), cotw));
					}
					IJV.push_back(Eigen::Triplet<Scalar, int>(v_it->idx(), v_it->idx(), w_ii));
					MTrip.push_back(Eigen::Triplet<Scalar, int>(v_it->idx(), v_it->idx(), 0.5f / area));
				}
				L.setFromTriplets(IJV.begin(), IJV.end());
				M.setFromTriplets(MTrip.begin(), MTrip.end());

				return true;
			}


			static float getAverageEdgeLength(Triangle_mesh &mesh)
			{
				float average_edge_length = 0.0f;
				for (Triangle_mesh::EdgeIter e_it = mesh.edges_begin(); e_it != mesh.edges_end(); e_it++)
					average_edge_length += mesh.calc_edge_length(*e_it);
				float edgeNum = (float)mesh.n_edges();
				average_edge_length /= edgeNum;
				return average_edge_length;
			}
			static bool MeshFair(TriMeshT& mesh, std::vector<VHandleT>& freevhs, int continus)
			{
				if (!mesh.is_trimesh())
					return false;
				Eigen::Matrix<double, Eigen::Dynamic, 3> b(freevhs.size(), 3);
				Eigen::Matrix<double, Eigen::Dynamic, 3> x(freevhs.size(), 3);
				b.setZero();
				Eigen::SparseMatrix<double>  L, Mat, M, LU;
				CotLaplaceMatrix(mesh, L, M);
				if (0 == continus)
				{
					Mat = M * L;
				}
				else if (1 == continus)
				{
					Mat = M * L* M *L;
				}
				else
				{
					Mat = M * L;
					Eigen::SparseMatrix<double> coff = Mat;
					for (int i = 0; i < continus; ++i) {
						Mat = Mat * coff;
					}
				}
				int id = 0;
				std::map<VertexHandle, int> idmap;
				for (auto vh : freevhs)
				{
					idmap[vh] = id;
					id++;
				}
				std::vector<Eigen::Triplet<double, int> > IJV;
				for (int k = 0; k < Mat.outerSize(); ++k)
				{
					// Iterate over inside
					for (Eigen::SparseMatrix<double>::InnerIterator it(Mat, k); it; ++it)
					{
						double v = it.value();
						VertexHandle rowvh = VertexHandle((int)it.row());
						VertexHandle colvh = VertexHandle((int)it.col());
						if (idmap.find(rowvh) == idmap.end())
							continue;
						if (idmap.find(colvh) != idmap.end())
						{
							IJV.push_back(Eigen::Triplet<double, int>(idmap[rowvh], idmap[colvh], v));
						}
						else
						{
							auto p = mesh.point(colvh);
							b(idmap[rowvh], 0) += -1.0f*v*p[0];
							b(idmap[rowvh], 1) += -1.0f*v*p[1];
							b(idmap[rowvh], 2) += -1.0f*v*p[2];
						}
					}
				}
				LU.resize(freevhs.size(), freevhs.size());
				LU.setFromTriplets(IJV.begin(), IJV.end());
				LU.makeCompressed();
				Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> >   solver;
				solver.compute(LU);
				if (solver.info() != Eigen::Success)
					return false;
				x = solver.solve(b);
				auto v_end = mesh.vertices_end();
				float ave_edge = 5 * getAverageEdgeLength(mesh);
				for (auto v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
				{
					if (idmap.find(*v_it) == idmap.end())
						continue;
					int index = idmap[*v_it];
					auto& p = mesh.point(*v_it);
					Triangle_mesh::Point ep;
					ep[0] = x(index, 0);
					ep[1] = x(index, 1);
					ep[2] = x(index, 2);
					float cfd = (p - ep).length();
					if (cfd >= ave_edge)
						return false;
				}
				for (auto v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
				{
					if (idmap.find(*v_it) == idmap.end())
						continue;
					int index = idmap[*v_it];
					auto& p = mesh.point(*v_it);
					p[0] = x(index, 0);
					p[1] = x(index, 1);
					p[2] = x(index, 2);
				}

				return true;
			}
		};
	}

}
//From CGAL
typedef struct Weight_min_max_dihedral_and_area
{
	template<class Weight_, class IsValid>
	friend struct Weight_calculator;

	template<class Weight_>
	friend struct Weight_incomplete;

	Weight_min_max_dihedral_and_area(Real angle, Real area, Real asp) : w(angle, area), aspect(asp) { }
	Weight_min_max_dihedral_and_area() : w(180, FLT_MAX) {}
	Real angle() const { return w.first; }
	Real area()  const { return w.second; }
	Real Aspect() const { return aspect; }

	Weight_min_max_dihedral_and_area operator+(
		const Weight_min_max_dihedral_and_area & _other) const {
		return Weight(std::max(w.first, _other.w.first),
			w.second + _other.w.second, std::min(aspect, _other.aspect));
	}
	bool operator<(const Weight_min_max_dihedral_and_area & w2) const {
		return (angle() < w2.angle() ||
			(angle() == w2.angle() && area() < w2.area()))/*&&
			(angle()==w2.angle()&&area()==w2.area()&&Aspect()<w2.Aspect()*/;
	}
	bool operator==(const Weight_min_max_dihedral_and_area& w2) const
	{
		return w.first == w2.w.first && w.second == w2.w.second/*&&aspect==w2.aspect*/;
	}
	bool operator!=(const Weight_min_max_dihedral_and_area& w2) const
	{
		return !(*this == w2);
	}
	static const Weight_min_max_dihedral_and_area DEFAULT()
	{
		return Weight_min_max_dihedral_and_area(0, 0, 0);
	}
	static const Weight_min_max_dihedral_and_area NOT_VALID()
	{
		return Weight_min_max_dihedral_and_area(-1, -1, -1);
	}
	std::pair<Real, Real> w;
	Real aspect;
}Weight;


static Real dihedral_angle(Triangle_mesh& mesh_,
	const VertexHandle& _u, const VertexHandle& _v,
	const VertexHandle& _a, const VertexHandle& _b)
{
	Point u(mesh_.point(_u));
	Point v(mesh_.point(_v));
	Point a(mesh_.point(_a));
	Point b(mesh_.point(_b));
	Point n0((v - u) % (a - v));
	Point n1((u - v) % (b - u));
	n0.normalize();
	n1.normalize();
	return static_cast<Real>(acos(n0 | n1) * 180.0 *INV_M_PI);
}

static Real area(Triangle_mesh& mesh_,
	const VertexHandle& _a, const VertexHandle& _b, const VertexHandle& _c)
{
	Point a(mesh_.point(_a));
	Point b(mesh_.point(_b));
	Point c(mesh_.point(_c));

	Point n((b - a) % (c - b));

	return static_cast<Real>(0.5 * n.norm());
}

static bool exists_edge(Triangle_mesh& mesh_,
	const VertexHandle& _u, const VertexHandle& _w)
{
	for (auto vohi = mesh_.voh_iter(_u); vohi.is_valid(); ++vohi)
		if (!mesh_.is_boundary(mesh_.edge_handle(*vohi)))
			if (mesh_.to_vertex_handle(*vohi) == _w)
				return true;
	return false;
}

static Real Aspect(Triangle_mesh& mesh_,
	const VertexHandle& _a, const VertexHandle& _b, const VertexHandle& _c) {
	Point a(mesh_.point(_a));
	Point b(mesh_.point(_b));
	Point c(mesh_.point(_c));

	Real ab = (a - b).length();
	Real bc = (b - c).length();
	Real ca = (c - a).length();

	Real minlen = std::min(ab, std::min(bc, ca));
	Real maxlen = std::max(ab, std::max(bc, ca));
	return 1 - minlen / maxlen;
}

static Weight weight(Triangle_mesh& mesh_,
	std::vector<VertexHandle>& boundary_vertex_,
	std::vector<VertexHandle>& opposite_vertex_,
	NMatrix<int>& l_,
	int _i, int _j, int _k)
{

	if (exists_edge(mesh_, boundary_vertex_[_i], boundary_vertex_[_j]) ||
		exists_edge(mesh_, boundary_vertex_[_j], boundary_vertex_[_k]) ||
		exists_edge(mesh_, boundary_vertex_[_k], boundary_vertex_[_i]))
		return Weight();

	if (l_(_i, _j) == -1) return Weight();
	if (l_(_j, _k) == -1) return Weight();


	Real angle = (Real)0.0f;

	if (_i + 1 == _j)
		angle = std::max(angle, dihedral_angle(mesh_, boundary_vertex_[_i],
			boundary_vertex_[_j],
			boundary_vertex_[_k],
			opposite_vertex_[_i]));
	else
		angle = std::max(angle, dihedral_angle(mesh_, boundary_vertex_[_i],
			boundary_vertex_[_j],
			boundary_vertex_[_k],
			boundary_vertex_[l_(_i, _j)]));

	if (_j + 1 == _k)
		angle = std::max(angle, dihedral_angle(mesh_, boundary_vertex_[_j],
			boundary_vertex_[_k],
			boundary_vertex_[_i],
			opposite_vertex_[_j]));
	else
		angle = std::max(angle, dihedral_angle(mesh_, boundary_vertex_[_j],
			boundary_vertex_[_k],
			boundary_vertex_[_i],
			boundary_vertex_[l_(_j, _k)]));
	if (_i == 0 && _k == (int)boundary_vertex_.size() - 1)
		angle = std::max(angle, dihedral_angle(mesh_, boundary_vertex_[_k],
			boundary_vertex_[_i],
			boundary_vertex_[_j],
			opposite_vertex_[_k]));
	return Weight(angle,
		area(mesh_, boundary_vertex_[_i],
			boundary_vertex_[_j],
			boundary_vertex_[_k]),
		Aspect(mesh_, boundary_vertex_[_i],
			boundary_vertex_[_j],
			boundary_vertex_[_k]));
}

static bool fill(Triangle_mesh& mesh_,
	std::vector<VertexHandle>& boundary_vertex_,
	NMatrix<int>& l_, std::vector< EdgeHandle >& hole_edge_,
	std::vector< FaceHandle >& hole_triangle_,
	int _i, int _j)
{

	if (_i + 1 == _j)
		return true;
	FaceHandle fh = mesh_.add_face(boundary_vertex_[_i],
		boundary_vertex_[l_(_i, _j)],
		boundary_vertex_[_j]);
	hole_triangle_.push_back(fh);
	if (!fh.is_valid())
		return false;
	hole_edge_.push_back(mesh_.edge_handle
	(mesh_.find_halfedge(boundary_vertex_[_i],
		boundary_vertex_[l_(_i, _j)])));
	hole_edge_.push_back(mesh_.edge_handle
	(mesh_.find_halfedge(boundary_vertex_[l_(_i, _j)],
		boundary_vertex_[_j])));
	if (!fill(mesh_, boundary_vertex_, l_, hole_edge_,
		hole_triangle_, _i, l_(_i, _j)) || !fill(mesh_, boundary_vertex_, l_, hole_edge_,
			hole_triangle_, l_(_i, _j), _j))
		return false;
	else
		return true;
}

void HoleFiller::SmoothMeshBoundary(Triangle_mesh& m_teeth)
{
	if (!m_teeth.has_edge_status())
		m_teeth.request_edge_status();
	if (!m_teeth.has_vertex_status())
		m_teeth.request_vertex_status();
	if (!m_teeth.has_face_status())
		m_teeth.request_face_status();
	Triangle_mesh& mesh = m_teeth;
	auto clearHoleFunc = [](Triangle_mesh& mesh)
	{
		bool isdegnerate = true;
		while (isdegnerate)
		{
			isdegnerate = false;
			auto v_end = mesh.vertices_end();
			Point point;
			for (auto v_it = mesh.vertices_sbegin(); v_it != v_end; ++v_it)
			{
				VertexHandle vh = *v_it;
				if (!mesh.is_boundary(*v_it))
					continue;
				point.vectorize(0.0f);
				int ivalence = 0;
				Point vhp = mesh.point(vh);
				std::vector<VertexHandle>  vcircle;
				for (auto vv_it = mesh.cvv_iter(vh); vv_it.is_valid(); ++vv_it)
				{
					if (mesh.is_boundary(*vv_it))
					{
						point += 0.5f*(vhp + mesh.point(*vv_it));
						ivalence++;
						vcircle.push_back(*vv_it);
					}
				}
				if (ivalence > 2)
				{
					mesh.delete_vertex(vh);
					isdegnerate = true;
				}
				else if (2 == ivalence)
				{
					HalfedgeHandle starthh0 = mesh.find_halfedge(vh, vcircle[0]);
					HalfedgeHandle starthh1 = mesh.opposite_halfedge_handle(starthh0);
					FaceHandle fh0;
					if (mesh.is_boundary(starthh0))
					{
						fh0 = mesh.face_handle(starthh1);
					}
					else
					{
						fh0 = mesh.face_handle(starthh0);
					}
					//
					HalfedgeHandle endh0 = mesh.find_halfedge(vh, vcircle[1]);
					HalfedgeHandle endh1 = mesh.opposite_halfedge_handle(endh0);
					FaceHandle fh1;
					if (mesh.is_boundary(endh0))
					{
						fh1 = mesh.face_handle(endh1);
					}
					else
					{
						fh1 = mesh.face_handle(endh0);
					}
					const Point& vp0 = mesh.point(vcircle[0]);
					const Point& vp1 = mesh.point(vcircle[1]);
					Point e0 = (vhp - vp0).normalized();
					Point e1 = (vhp - vp1).normalized();
					float dote01 = (e0 | e1);
					if (dote01 > -0.3090f)
					{
						point *= (1.0f / (1.0f*ivalence));
						mesh.set_point(vh, point);
					}
				}
				else
				{
					isdegnerate = true;
					mesh.delete_vertex(vh);
				}
			}
		}
		mesh.garbage_collection();
	};

	//删除孤立顶点
	mesh.delete_isolated_vertices();
	mesh.garbage_collection();

	//删除非流行顶点
	while (true)
	{
		std::vector<VertexHandle>  iosvh;
		auto v_end = mesh.vertices_end();
		for (auto v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			if (!mesh.is_manifold(*v_it))
				iosvh.push_back(*v_it);
		}
		for (auto& vh : iosvh)
		{
			mesh.delete_vertex(vh);
		}
		if (0 == iosvh.size())
			break;
		mesh.garbage_collection();
	}

	HalfedgeHandle bhh;
	auto fh_end = mesh.halfedges_end();
	for (auto fh_it = mesh.halfedges_begin(); fh_it != fh_end; ++fh_it)
	{
		if (mesh.is_boundary(*fh_it))
		{
			bhh = *fh_it;
			break;
		}
	}
	float maxangle = 0.523599f;//30
	if (!mesh.is_boundary(bhh))
	{
		return;
	}
	HalfedgeHandle current = bhh;
	HalfedgeHandle breakhh = bhh;

	while (true)
	{
		bool newface = false;
		if (!mesh.is_boundary(current))
		{
			break;
		}
		HalfedgeHandle nexthh = mesh.next_halfedge_handle(current);
		HalfedgeHandle prehh = mesh.prev_halfedge_handle(current);
		VertexHandle vh0 = mesh.to_vertex_handle(current);
		VertexHandle vh1 = mesh.to_vertex_handle(prehh);
		VertexHandle vh2 = mesh.to_vertex_handle(nexthh);
		const Point& p0 = mesh.point(vh0);
		const Point& p1 = mesh.point(vh1);
		const Point& p2 = mesh.point(vh2);
		Point e1 = p0 - p1;
		Point e2 = p2 - p1;
		float v0angle = (e1 | e2) / (e1.length()*e2.length());
		if (v0angle < -1.0f)
			v0angle = -1.0f;
		if (v0angle > 1.0f)
			v0angle = 1.0f;
		v0angle = acos(v0angle);
		if (v0angle < 0.3f)
		{
			current = nexthh;
			if ((current == breakhh) && (!newface))
				break;
			continue;
		}
		FaceHandle currentfh = mesh.face_handle(mesh.opposite_halfedge_handle(current));
		FaceHandle nextfh = mesh.face_handle(mesh.opposite_halfedge_handle(nexthh));
		Point nor0 = (e1%e2).normalized();
		Point nor1 = mesh.calc_face_normal(currentfh);
		Point nor2 = mesh.calc_face_normal(nextfh);
		float v1dot = (nor0 | nor1) / (nor0.length()*nor1.length());
		float v2dot = (nor0 | nor2) / (nor0.length()*nor2.length());
		if (v1dot < -1.0f)
			v1dot = -1.0f;
		if (v2dot > 1.0f)
			v2dot = 1.0f;
		float v1angle = acos(v1dot);
		float v2angle = acos(v2dot);
		if ((v1angle > maxangle) || (v2angle > maxangle))
		{
			current = nexthh;
			if ((current == breakhh) && (!newface))
				break;
			continue;
		}
		//
		FaceHandle fh = mesh.add_face(vh1, vh0, vh2);
		if (!fh.is_valid())
			break;
		current = prehh;
		breakhh = prehh;
		newface = true;
	}
	clearHoleFunc(mesh);

	m_teeth.release_face_status();
	m_teeth.release_edge_status();
	m_teeth.release_vertex_status();
}


static bool refineMesh(Triangle_mesh& mesh, std::vector<VertexHandle>& boundary, Real avg)
{
	using namespace std;
	using ftype = Real;
	ftype avelen = avg;
	ftype angle = 17;
	using Vertex3D = Triangle_mesh::Point;
	mesh.request_edge_status();
	mesh.request_face_status();
	mesh.request_vertex_status();
	mesh.request_face_normals();
	mesh.request_vertex_normals();
	mesh.update_normals();
	OpenMesh::VPropHandleT<double>  mVertScale;
	mesh.add_property(mVertScale, "scale");
	Triangle_mesh::VertexOHalfedgeIter voh_iter;
	auto v_end = mesh.vertices_end();
	ftype msize = static_cast<ftype>(avelen);
	for (auto v_it = mesh.vertices_begin(); v_it != v_end; ++v_it) {
		if (mesh.is_boundary(*v_it)) {
			mesh.status(*v_it).set_locked(true);
			double total = 0.0;
			int neiborsize = 0;
			voh_iter = mesh.voh_iter(*v_it);
			const auto& v = mesh.point(*v_it);
			for (; voh_iter.is_valid(); ++voh_iter)
			{
				const auto& p = mesh.point(mesh.to_vertex_handle(*voh_iter));
				total += (p - v).length();
				neiborsize++;
			}
			mesh.property(mVertScale, *v_it) = msize;
		}
	}
	vector<EdgeHandle> mBoundaryEH;
	vector<FaceHandle> mNewFillFace;
	auto e_end = mesh.edges_end();
	for (auto e_it = mesh.edges_begin(); e_it != e_end; ++e_it)
		if (mesh.is_boundary(*e_it))
			mesh.status(*e_it).set_locked(true);
	auto f_end = mesh.faces_end();
	mNewFillFace.reserve(mesh.n_faces());
	for (auto f_it = mesh.faces_begin(); f_it != f_end; ++f_it) {
		mNewFillFace.push_back(*f_it);
	}
	ftype  alpha = static_cast<ftype>(std::sqrtf(angle));
	auto InCircumsphere = [](
		const Vertex3D & x,
		const Vertex3D & a,
		const Vertex3D & b,
		const Vertex3D & c) {
		auto ab = b - a;
		auto ac = c - a;
		double a00 = -2.0f * (ab | a);
		double a01 = -2.0f * (ab | b);
		double a02 = -2.0f * (ab | c);
		double b0 = a.sqrnorm() - b.sqrnorm();
		double a10 = -2.0f * (ac | a);
		double a11 = -2.0f * (ac | b);
		double a12 = -2.0f * (ac | c);
		double b1 = a.sqrnorm() - c.sqrnorm();
		double alpha = -(-a11 * a02 + a01 * a12 - a12 * b0 + b1 * a02 + a11 * b0 - a01 * b1)
			/ (-a11 * a00 + a11 * a02 - a10 * a02 + a00 * a12 + a01 * a10 - a01 * a12);
		double beta = (a10*b0 - a10 * a02 - a12 * b0 + a00 * a12 + b1 * a02 - a00 * b1)
			/ (-a11 * a00 + a11 * a02 - a10 * a02 + a00 * a12 + a01 * a10 - a01 * a12);
		double gamma = (-a11 * a00 - a10 * b0 + a00 * b1 + a11 * b0 + a01 * a10 - a01 * b1)
			/ (-a11 * a00 + a11 * a02 - a10 * a02 + a00 * a12 + a01 * a10 - a01 * a12);
		auto center = alpha * a + beta * b + gamma * c;
		return (x - center).sqrnorm() < (a - center).sqrnorm();
	};
	auto Relax = [&InCircumsphere](EdgeHandle eh, Triangle_mesh& mHoleMesh) {
		if (mHoleMesh.status(eh).locked())
			return false;
		HalfedgeHandle h0 = mHoleMesh.halfedge_handle(eh, 0);
		HalfedgeHandle h1 = mHoleMesh.halfedge_handle(eh, 1);
		auto u(mHoleMesh.point(mHoleMesh.to_vertex_handle(h0)));
		auto v(mHoleMesh.point(mHoleMesh.to_vertex_handle(h1)));
		auto a(mHoleMesh.point(mHoleMesh.to_vertex_handle(mHoleMesh.next_halfedge_handle(h0))));
		auto b(mHoleMesh.point(mHoleMesh.to_vertex_handle(mHoleMesh.next_halfedge_handle(h1))));
		if (InCircumsphere(a, u, v, b) || InCircumsphere(b, u, v, a)) {
			if (mHoleMesh.is_flip_ok(eh)) {
				mHoleMesh.flip(eh);
				return true;
			}
			else
				mHoleMesh.status(eh).set_selected(true);
		}
		return false;
	};
	auto Subdivide = [&alpha, &mesh, &mNewFillFace, &mVertScale, &mBoundaryEH, &Relax]() {
		bool status = false;
		size_t facenum = mNewFillFace.size();
		for (size_t i = 0; i < facenum; ++i) {
			HalfedgeHandle hh = mesh.halfedge_handle(mNewFillFace[i]);
			VertexHandle vi = mesh.to_vertex_handle(hh);
			VertexHandle vj = mesh.to_vertex_handle(mesh.prev_halfedge_handle(hh));
			VertexHandle vk = mesh.to_vertex_handle(mesh.next_halfedge_handle(hh));
			const auto& vip = mesh.point(vi);
			const auto& vjp = mesh.point(vj);
			const auto& vkp = mesh.point(vk);
			auto c = (vip + vjp + vkp) / 3.0f;
			const double vis = mesh.property(mVertScale, vi);
			const double vjs = mesh.property(mVertScale, vj);
			const double vks = mesh.property(mVertScale, vk);
			double sac = (vis + vjs + vks) / 3.0f;
			double dist_c_vi = (c - vip).length();
			double dist_c_vj = (c - vjp).length();
			double dist_c_vk = (c - vkp).length();
			if ((dist_c_vi + dist_c_vj + dist_c_vk) / 3.0f < sac)
				continue;
			if ((alpha * dist_c_vi > sac) &&
				(alpha * dist_c_vj > sac) &&
				(alpha * dist_c_vk > sac) &&
				(alpha * dist_c_vi > vis) &&
				(alpha * dist_c_vj > vjs) &&
				(alpha * dist_c_vk > vks)) {
				VertexHandle ch = mesh.add_vertex(c);
				mesh.split(mNewFillFace[i], ch);
				for (auto vfi = mesh.vf_iter(ch); vfi.is_valid(); ++vfi)
					if (*vfi != mNewFillFace[i])
						mNewFillFace.push_back(*vfi);
				for (auto vei = mesh.ve_iter(ch); vei.is_valid(); ++vei)
					mBoundaryEH.push_back(*vei);
				auto fei = mesh.fe_iter(mNewFillFace[i]);
				EdgeHandle  e0 = *fei; ++fei;
				EdgeHandle  e1 = *fei; ++fei;
				EdgeHandle  e2 = *fei; ++fei;
				Relax(e0, mesh);
				Relax(e1, mesh);
				Relax(e2, mesh);
				mesh.property(mVertScale, ch) = sac;
				status = true;
			}
		}
		return status;
	};
	auto CRelax = [&mBoundaryEH, &Relax](Triangle_mesh& mHoleMesh) {
		bool status = false;
		for (size_t i = 0; i < mBoundaryEH.size(); ++i)
			if (Relax(mBoundaryEH[i], mHoleMesh))
				status = true;
		return status;
	};
	for (int i = 0; i < 10; ++i) {
		bool is_subdivided = Subdivide();
		if (!is_subdivided)
			break;
		bool is_relaxed = CRelax(mesh);
		if (!is_relaxed)
			break;
	}
	for (auto e_it = mesh.edges_begin(); e_it != e_end; ++e_it)
		if (mesh.is_boundary(*e_it))
			mesh.status(*e_it).set_locked(false);
	mesh.remove_property(mVertScale);
	mesh.release_edge_status();
	mesh.release_face_status();
	mesh.release_vertex_status();
	mesh.release_face_normals();
	mesh.release_vertex_normals();
	return true;

}


static void selectRRV(
	Triangle_mesh& mesh, const std::vector<VertexHandle>& vhs,
	int n, std::vector<VertexHandle>& pts)
{
	std::vector<bool> iscall(mesh.n_vertices(), false);
	for (auto ip : vhs)
		iscall[ip.idx()] = true;
	auto nvhs = vhs;
	int count = -1;
	pts.clear();
	pts.insert(pts.end(), vhs.begin(), vhs.end());
	while (++count < n) {
		std::vector<VertexHandle> apts;
		for (auto ip : nvhs) {
			for (auto vv = mesh.vv_begin(ip); vv != mesh.vv_end(ip); ++vv) {
				if (iscall[vv->idx()] == true)
					continue;
				apts.push_back(*vv);
				iscall[vv->idx()] = true;
			}
		}
		pts.insert(pts.end(), apts.begin(), apts.end());
		nvhs.swap(apts);
	}
}



static void removeDegeneratedFaces(Triangle_mesh& mesh_, std::vector< FaceHandle >& _faceHandles)
{
	for (int i = _faceHandles.size() - 1; i >= 0; i--) {
		if (mesh_.status(_faceHandles[i]).deleted()) {
			_faceHandles.erase(_faceHandles.begin() + i);
			continue;
		}
		auto fvi = mesh_.fv_iter(_faceHandles[i]);
		Point v0 = mesh_.point(*fvi);
		++fvi;
		Point v1 = mesh_.point(*fvi);
		++fvi;
		Point v2 = mesh_.point(*fvi);
		Point v0v1 = v1 - v0;
		Point v0v2 = v2 - v0;
		Point n = v0v1 % v0v2;
		Real d = n.sqrnorm();
		if (d < FLT_MIN && d > -FLT_MIN) {
			auto hIt = mesh_.fh_iter(_faceHandles[i]);
			while (hIt.is_valid()) {
				if (mesh_.is_collapse_ok(*hIt)) {
					mesh_.collapse(*hIt);
					_faceHandles.erase(_faceHandles.begin() + i);
					break;
				}
				else {
					++hIt;
				}
			}
		}
	}
}

static Real calc_avg_length(Triangle_mesh& mesh) {
	Real avg = 0;
	for (auto ie : mesh.edges()) {
		avg += mesh.calc_edge_length(ie);
	}
	avg /= (Real)mesh.n_edges();
	return avg;
}


bool HoleFiller::hole_fillC1(Triangle_mesh& mesh, HalfedgeHandle& hh, int _stages)
{
	if (!mesh.is_boundary(hh)) {
		return false;
	}

	if (!mesh.has_edge_status())
		mesh.request_edge_status();
	if (!mesh.has_face_status())
		mesh.request_face_status();
	if (!mesh.has_vertex_status())
		mesh.request_vertex_status();
	Triangle_mesh::VertexHandle old_last_handle = *(--mesh.vertices_end());

	std::vector<VertexHandle> boundary_vertex_;
	std::vector<VertexHandle> opposite_vertex_;
	HalfedgeHandle ch = hh;
	do {
		boundary_vertex_.push_back(mesh.from_vertex_handle(ch));
		opposite_vertex_.push_back(mesh.to_vertex_handle
		(mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(ch))));
		int c = 0;
		VertexHandle vh = mesh.to_vertex_handle(ch);
		for (Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, vh); voh_it.is_valid(); ++voh_it)
			if (mesh.is_boundary(*voh_it))
				c++;
		if (c >= 2) {
			HalfedgeHandle  op = mesh.opposite_halfedge_handle(ch);
			typename Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, op);
			ch = *(++voh_it);
		}
		else
			ch = mesh.next_halfedge_handle(ch);
	} while (ch != hh);
	int nv = (int)boundary_vertex_.size();
	NMatrix<Weight> w_(nv, nv);
	NMatrix<int> l_(nv, nv, 0);
	for (int i = 0; i < nv - 1; ++i)
		w_(i, i + 1) = Weight(0, 0, 0);
	for (int j = 2; j < nv; ++j)
	{
#pragma omp parallel for shared(j)
		for (int i = 0; i < nv - j; ++i)
		{
			Weight valmin;
			int   argmin = -1;
			for (int m = i + 1; m < i + j; ++m)
			{
				Weight newval = w_(i, m) + w_(m, i + j) + weight(mesh,
					boundary_vertex_, opposite_vertex_, l_, i, m, i + j);
				if (newval < valmin)
				{
					valmin = newval;
					argmin = m;
				}
			}
			w_(i, i + j) = valmin;
			l_(i, i + j) = argmin;
		}
	}
	std::vector< EdgeHandle > hole_edge_;
	std::vector< FaceHandle > hole_triangle_;
	Triangle_mesh hole_mesh;
	std::vector<VertexHandle> boundary;
	boundary.reserve(boundary_vertex_.size());
	for (auto& ib : boundary_vertex_) {
		boundary.emplace_back(hole_mesh.add_vertex(mesh.point(ib)));
	}


	if (fill(hole_mesh, boundary, l_, hole_edge_,
		hole_triangle_, 0, nv - 1) && boundary_vertex_.size() > 10) {
		if (_stages <= 1)
			return true;
		Triangle_mesh chole = hole_mesh;

		Real avge = calc_avg_length(mesh);

		if (refineMesh(hole_mesh, boundary, avge))
		{
			int nbound = (int)boundary.size();
			int nhole = (int)hole_mesh.n_vertices();
			auto points = hole_mesh.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : hole_mesh.faces()) {
				auto fv = hole_mesh.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
			Fair::SmoothMesh meshfair;
			std::vector<VertexHandle> fairvhs;
			selectRRV(mesh, boundary_vertex_, 2, fairvhs);
			for (auto iv : mesh.vertices()) {
				mesh.status(iv).set_locked(true);
			}
			for (auto iv : boundary_vertex_) {
				mesh.status(iv).set_locked(false);
			}
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Continuity
				continuity = OpenMesh::Smoother::SmootherT<Triangle_mesh>::C1;
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Component
				component = OpenMesh::Smoother::SmootherT<Triangle_mesh>::Tangential_and_Normal;
			OpenMesh::Smoother::JacobiLaplaceSmootherT<Triangle_mesh> smoother(mesh);
			smoother.initialize(component, continuity);
			smoother.smooth(5);
			if (!meshfair.MeshFair(mesh, fairvhs, 1))
				std::cout << "poisson fail!" << std::endl;
		}
		else {
			int nbound = (int)boundary.size();
			int nhole = (int)chole.n_vertices();
			auto points = chole.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : chole.faces()) {
				auto fv = chole.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
		}
	}
	else
	{
		int nbound = (int)boundary.size();
		int nhole = (int)hole_mesh.n_vertices();
		auto points = hole_mesh.points();
		for (int i = nbound; i < nhole; ++i) {
			boundary_vertex_.push_back(mesh.add_vertex(points[i]));
		}
		for (auto f : hole_mesh.faces()) {
			auto fv = hole_mesh.fv_begin(f);
			int v0 = fv->idx(); ++fv;
			int v1 = fv->idx(); ++fv;
			int v2 = fv->idx();
			mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
		}
	}
	removeDegeneratedFaces(mesh, hole_triangle_);
	mesh.release_edge_status();
	mesh.release_face_status();
	mesh.release_vertex_status();
	return true;
}

bool HoleFiller::hole_fillC0(Triangle_mesh& mesh, HalfedgeHandle& hh, int _stages)
{
	if (!mesh.is_boundary(hh)) {
		return false;
	}

	if (!mesh.has_edge_status())
		mesh.request_edge_status();
	if (!mesh.has_face_status())
		mesh.request_face_status();
	if (!mesh.has_vertex_status())
		mesh.request_vertex_status();
	Triangle_mesh::VertexHandle old_last_handle = *(--mesh.vertices_end());

	std::vector<VertexHandle> boundary_vertex_;
	std::vector<VertexHandle> opposite_vertex_;
	HalfedgeHandle ch = hh;
	do {
		boundary_vertex_.push_back(mesh.from_vertex_handle(ch));
		opposite_vertex_.push_back(mesh.to_vertex_handle
		(mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(ch))));
		int c = 0;
		VertexHandle vh = mesh.to_vertex_handle(ch);
		for (Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, vh); voh_it.is_valid(); ++voh_it)
			if (mesh.is_boundary(*voh_it))
				c++;
		if (c >= 2) {
			HalfedgeHandle  op = mesh.opposite_halfedge_handle(ch);
			typename Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, op);
			ch = *(++voh_it);
		}
		else
			ch = mesh.next_halfedge_handle(ch);
	} while (ch != hh);
	int nv = (int)boundary_vertex_.size();
	NMatrix<Weight> w_(nv, nv);
	NMatrix<int> l_(nv, nv, 0);
	for (int i = 0; i < nv - 1; ++i)
		w_(i, i + 1) = Weight(0, 0, 0);
	for (int j = 2; j < nv; ++j)
	{
#pragma omp parallel for shared(j)
		for (int i = 0; i < nv - j; ++i)
		{
			Weight valmin;
			int   argmin = -1;
			for (int m = i + 1; m < i + j; ++m)
			{
				Weight newval = w_(i, m) + w_(m, i + j) + weight(mesh,
					boundary_vertex_, opposite_vertex_, l_, i, m, i + j);
				if (newval < valmin)
				{
					valmin = newval;
					argmin = m;
				}
			}
			w_(i, i + j) = valmin;
			l_(i, i + j) = argmin;
		}
	}
	std::vector< EdgeHandle > hole_edge_;
	std::vector< FaceHandle > hole_triangle_;
	Triangle_mesh hole_mesh;
	std::vector<VertexHandle> boundary;
	boundary.reserve(boundary_vertex_.size());
	for (auto& ib : boundary_vertex_) {
		boundary.emplace_back(hole_mesh.add_vertex(mesh.point(ib)));
	}


	if (fill(hole_mesh, boundary, l_, hole_edge_,
		hole_triangle_, 0, nv - 1) && boundary_vertex_.size() > 10) {
		if (_stages <= 1)
			return true;
		Triangle_mesh chole = hole_mesh;

		Real avge = calc_avg_length(mesh);

		if (refineMesh(hole_mesh, boundary, avge))
		{
			int nbound = (int)boundary.size();
			int nhole = (int)hole_mesh.n_vertices();
			auto points = hole_mesh.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : hole_mesh.faces()) {
				auto fv = hole_mesh.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
			Fair::SmoothMesh meshfair;
			std::vector<VertexHandle> fairvhs;
			selectRRV(mesh, boundary_vertex_, 2, fairvhs);
			for (auto iv : mesh.vertices()) {
				mesh.status(iv).set_locked(true);
			}
			for (auto iv : boundary_vertex_) {
				mesh.status(iv).set_locked(false);
			}
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Continuity
				continuity = OpenMesh::Smoother::SmootherT<Triangle_mesh>::C1;
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Component
				component = OpenMesh::Smoother::SmootherT<Triangle_mesh>::Tangential_and_Normal;
			OpenMesh::Smoother::JacobiLaplaceSmootherT<Triangle_mesh> smoother(mesh);
			smoother.initialize(component, continuity);
			smoother.smooth(5);
			if (!meshfair.MeshFair(mesh, fairvhs, 0))
				std::cout << "poisson fail!" << std::endl;
		}
		else {
			int nbound = (int)boundary.size();
			int nhole = (int)chole.n_vertices();
			auto points = chole.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : chole.faces()) {
				auto fv = chole.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
		}
	}
	else
	{
		int nbound = (int)boundary.size();
		int nhole = (int)hole_mesh.n_vertices();
		auto points = hole_mesh.points();
		for (int i = nbound; i < nhole; ++i) {
			boundary_vertex_.push_back(mesh.add_vertex(points[i]));
		}
		for (auto f : hole_mesh.faces()) {
			auto fv = hole_mesh.fv_begin(f);
			int v0 = fv->idx(); ++fv;
			int v1 = fv->idx(); ++fv;
			int v2 = fv->idx();
			mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
		}
	}
	removeDegeneratedFaces(mesh, hole_triangle_);
	mesh.release_edge_status();
	mesh.release_face_status();
	mesh.release_vertex_status();
	return true;
}


bool HoleFiller::hole_fillC2(Triangle_mesh& mesh, HalfedgeHandle& hh, int _stages)
{
	if (!mesh.is_boundary(hh)) {
		return false;
	}

	if (!mesh.has_edge_status())
		mesh.request_edge_status();
	if (!mesh.has_face_status())
		mesh.request_face_status();
	if (!mesh.has_vertex_status())
		mesh.request_vertex_status();
	Triangle_mesh::VertexHandle old_last_handle = *(--mesh.vertices_end());

	std::vector<VertexHandle> boundary_vertex_;
	std::vector<VertexHandle> opposite_vertex_;
	HalfedgeHandle ch = hh;
	do {
		boundary_vertex_.push_back(mesh.from_vertex_handle(ch));
		opposite_vertex_.push_back(mesh.to_vertex_handle
		(mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(ch))));
		int c = 0;
		VertexHandle vh = mesh.to_vertex_handle(ch);
		for (Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, vh); voh_it.is_valid(); ++voh_it)
			if (mesh.is_boundary(*voh_it))
				c++;
		if (c >= 2) {
			HalfedgeHandle  op = mesh.opposite_halfedge_handle(ch);
			typename Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, op);
			ch = *(++voh_it);
		}
		else
			ch = mesh.next_halfedge_handle(ch);
	} while (ch != hh);
	int nv = (int)boundary_vertex_.size();
	NMatrix<Weight> w_(nv, nv);
	NMatrix<int> l_(nv, nv, 0);
	for (int i = 0; i < nv - 1; ++i)
		w_(i, i + 1) = Weight(0, 0, 0);
	for (int j = 2; j < nv; ++j)
	{
#pragma omp parallel for shared(j)
		for (int i = 0; i < nv - j; ++i)
		{
			Weight valmin;
			int   argmin = -1;
			for (int m = i + 1; m < i + j; ++m)
			{
				Weight newval = w_(i, m) + w_(m, i + j) + weight(mesh,
					boundary_vertex_, opposite_vertex_, l_, i, m, i + j);
				if (newval < valmin)
				{
					valmin = newval;
					argmin = m;
				}
			}
			w_(i, i + j) = valmin;
			l_(i, i + j) = argmin;
		}
	}
	std::vector< EdgeHandle > hole_edge_;
	std::vector< FaceHandle > hole_triangle_;
	Triangle_mesh hole_mesh;
	std::vector<VertexHandle> boundary;
	boundary.reserve(boundary_vertex_.size());
	for (auto& ib : boundary_vertex_) {
		boundary.emplace_back(hole_mesh.add_vertex(mesh.point(ib)));
	}


	if (fill(hole_mesh, boundary, l_, hole_edge_,
		hole_triangle_, 0, nv - 1) && boundary_vertex_.size() > 10) {
		if (_stages <= 1)
			return true;
		Triangle_mesh chole = hole_mesh;

		Real avge = calc_avg_length(mesh);

		if (refineMesh(hole_mesh, boundary, avge))
		{
			int nbound = (int)boundary.size();
			int nhole = (int)hole_mesh.n_vertices();
			auto points = hole_mesh.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : hole_mesh.faces()) {
				auto fv = hole_mesh.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
			Fair::SmoothMesh meshfair;
			std::vector<VertexHandle> fairvhs;
			selectRRV(mesh, boundary_vertex_, 2, fairvhs);
			for (auto iv : mesh.vertices()) {
				mesh.status(iv).set_locked(true);
			}
			for (auto iv : boundary_vertex_) {
				mesh.status(iv).set_locked(false);
			}
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Continuity
				continuity = OpenMesh::Smoother::SmootherT<Triangle_mesh>::C1;
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Component
				component = OpenMesh::Smoother::SmootherT<Triangle_mesh>::Tangential_and_Normal;
			OpenMesh::Smoother::JacobiLaplaceSmootherT<Triangle_mesh> smoother(mesh);
			smoother.initialize(component, continuity);
			smoother.smooth(5);
			if (!meshfair.MeshFair(mesh, fairvhs, 2))
				std::cout << "poisson fail!" << std::endl;
		}
		else {
			int nbound = (int)boundary.size();
			int nhole = (int)chole.n_vertices();
			auto points = chole.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : chole.faces()) {
				auto fv = chole.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
		}
	}
	else
	{
		int nbound = (int)boundary.size();
		int nhole = (int)hole_mesh.n_vertices();
		auto points = hole_mesh.points();
		for (int i = nbound; i < nhole; ++i) {
			boundary_vertex_.push_back(mesh.add_vertex(points[i]));
		}
		for (auto f : hole_mesh.faces()) {
			auto fv = hole_mesh.fv_begin(f);
			int v0 = fv->idx(); ++fv;
			int v1 = fv->idx(); ++fv;
			int v2 = fv->idx();
			mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
		}
	}
	removeDegeneratedFaces(mesh, hole_triangle_);
	mesh.release_edge_status();
	mesh.release_face_status();
	mesh.release_vertex_status();
	return true;
}

bool HoleFiller::hole_fillCK(Triangle_mesh& mesh, HalfedgeHandle& hh, int _stages)
{
	if (!mesh.is_boundary(hh)) {
		return false;
	}

	if (!mesh.has_edge_status())
		mesh.request_edge_status();
	if (!mesh.has_face_status())
		mesh.request_face_status();
	if (!mesh.has_vertex_status())
		mesh.request_vertex_status();
	Triangle_mesh::VertexHandle old_last_handle = *(--mesh.vertices_end());

	std::vector<VertexHandle> boundary_vertex_;
	std::vector<VertexHandle> opposite_vertex_;
	HalfedgeHandle ch = hh;
	do {
		boundary_vertex_.push_back(mesh.from_vertex_handle(ch));
		opposite_vertex_.push_back(mesh.to_vertex_handle
		(mesh.next_halfedge_handle(mesh.opposite_halfedge_handle(ch))));
		int c = 0;
		VertexHandle vh = mesh.to_vertex_handle(ch);
		for (Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, vh); voh_it.is_valid(); ++voh_it)
			if (mesh.is_boundary(*voh_it))
				c++;
		if (c >= 2) {
			HalfedgeHandle  op = mesh.opposite_halfedge_handle(ch);
			typename Triangle_mesh::VertexOHalfedgeIter voh_it(mesh, op);
			ch = *(++voh_it);
		}
		else
			ch = mesh.next_halfedge_handle(ch);
	} while (ch != hh);
	int nv = (int)boundary_vertex_.size();
	NMatrix<Weight> w_(nv, nv);
	NMatrix<int> l_(nv, nv, 0);
	for (int i = 0; i < nv - 1; ++i)
		w_(i, i + 1) = Weight(0, 0, 0);
	for (int j = 2; j < nv; ++j)
	{
#pragma omp parallel for shared(j)
		for (int i = 0; i < nv - j; ++i)
		{
			Weight valmin;
			int   argmin = -1;
			for (int m = i + 1; m < i + j; ++m)
			{
				Weight newval = w_(i, m) + w_(m, i + j) + weight(mesh,
					boundary_vertex_, opposite_vertex_, l_, i, m, i + j);
				if (newval < valmin)
				{
					valmin = newval;
					argmin = m;
				}
			}
			w_(i, i + j) = valmin;
			l_(i, i + j) = argmin;
		}
	}
	std::vector< EdgeHandle > hole_edge_;
	std::vector< FaceHandle > hole_triangle_;
	Triangle_mesh hole_mesh;
	std::vector<VertexHandle> boundary;
	boundary.reserve(boundary_vertex_.size());
	for (auto& ib : boundary_vertex_) {
		boundary.emplace_back(hole_mesh.add_vertex(mesh.point(ib)));
	}


	if (fill(hole_mesh, boundary, l_, hole_edge_,
		hole_triangle_, 0, nv - 1) && boundary_vertex_.size() > 10) {
		if (_stages <= 1)
			return true;
		Triangle_mesh chole = hole_mesh;

		Real avge = calc_avg_length(mesh);

		if (refineMesh(hole_mesh, boundary, avge))
		{
			int nbound = (int)boundary.size();
			int nhole = (int)hole_mesh.n_vertices();
			auto points = hole_mesh.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : hole_mesh.faces()) {
				auto fv = hole_mesh.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
			Fair::SmoothMesh meshfair;
			std::vector<VertexHandle> fairvhs;
			selectRRV(mesh, boundary_vertex_, 2, fairvhs);
			for (auto iv : mesh.vertices()) {
				mesh.status(iv).set_locked(true);
			}
			for (auto iv : boundary_vertex_) {
				mesh.status(iv).set_locked(false);
			}
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Continuity
				continuity = OpenMesh::Smoother::SmootherT<Triangle_mesh>::C1;
			OpenMesh::Smoother::SmootherT<Triangle_mesh>::Component
				component = OpenMesh::Smoother::SmootherT<Triangle_mesh>::Tangential_and_Normal;
			OpenMesh::Smoother::JacobiLaplaceSmootherT<Triangle_mesh> smoother(mesh);
			smoother.initialize(component, continuity);
			smoother.smooth(5);
			if (!meshfair.MeshFair(mesh, fairvhs, _stages))
				std::cout << "poisson fail!" << std::endl;
		}
		else {
			int nbound = (int)boundary.size();
			int nhole = (int)chole.n_vertices();
			auto points = chole.points();
			for (int i = nbound; i < nhole; ++i) {
				boundary_vertex_.push_back(mesh.add_vertex(points[i]));
			}
			for (auto f : chole.faces()) {
				auto fv = chole.fv_begin(f);
				int v0 = fv->idx(); ++fv;
				int v1 = fv->idx(); ++fv;
				int v2 = fv->idx();
				mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
			}
		}
	}
	else
	{
		int nbound = (int)boundary.size();
		int nhole = (int)hole_mesh.n_vertices();
		auto points = hole_mesh.points();
		for (int i = nbound; i < nhole; ++i) {
			boundary_vertex_.push_back(mesh.add_vertex(points[i]));
		}
		for (auto f : hole_mesh.faces()) {
			auto fv = hole_mesh.fv_begin(f);
			int v0 = fv->idx(); ++fv;
			int v1 = fv->idx(); ++fv;
			int v2 = fv->idx();
			mesh.add_face(boundary_vertex_[v0], boundary_vertex_[v1], boundary_vertex_[v2]);
		}
	}
	removeDegeneratedFaces(mesh, hole_triangle_);
	mesh.release_edge_status();
	mesh.release_face_status();
	mesh.release_vertex_status();
	return true;
}


#pragma warning(pop)