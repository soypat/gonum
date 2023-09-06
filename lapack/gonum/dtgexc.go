// Copyright ©2023 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package gonum

// Dtgexc reorders the generalized real Schur decomposition of a real
// matrix pair (A,B) using an orthogonal equivalence transformation
//
//	(A, B) = Q * (A, B) * Zᵀ,
//
// so that the diagonal block of (A, B) with row index IFST is moved
// to row ILST.
//
// (A, B) must be in generalized real Schur canonical form (as returned
// by DGGES), i.e. A is block upper triangular with 1×1 and 2×2
// diagonal blocks. B is upper triangular.
//
// Optionally, the matrices Q and Z of generalized Schur vectors are
// updated.
//
//	Q(in) * A(in) * Z(in)ᵀ = Q(out) * A(out) * Z(out)ᵀ
//	Q(in) * B(in) * Z(in)ᵀ = Q(out) * B(out) * Z(out)ᵀ
//
// Parameters:
//   - Output bool `illConditioned` is true if the transformed matrix pair (A,B)
//     would be too far from generalized Schur form; (A,B) and (Q,Z) left unchanged.
//   - Output bool `lworkTooSmall` is true if the given work slice is too small. Appropiate value
//     for lwork is stored in work[0].
//   - ifst, ilst specify the reordering of the diagonal blocks of (A,B).
//     The block with row index ifst is moved to row ilst, by a sequence of
//     swapping between adjacent blocks.
//   - work is of length at least max( 1, 4*n+16 )
//   - wantq, wantz indicate whether to update Q or Z transformation matrices, respectively.
//
// Dtgexc is an internal routine. It is exported for testing purposes only.
func (impl Implementation) Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, isWorkspaceQuery bool) (ifstOut, ilstOut int, illConditioned, lworkTooSmall bool) {
	lwmin := 1
	lwork := len(work)
	switch {
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case ldq < 1 || (wantq && ldq < max(1, n)):
		panic(badLdQ)
	case ldz < 1 || (wantz && ldz < max(1, n)):
		panic(badLdZ)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case !isWorkspaceQuery && n != 0 && lwork < max(1, 4*n+16):
		lworkTooSmall = true
		return ifst, ilst, false, lworkTooSmall
	case isWorkspaceQuery:
		if n > 0 {
			lwmin = max(1, 4*n+16)
		}
		work[0] = float64(lwmin)
		return ifst, ilst, false, false
	case n == 0:
		return ifst, ilst, false, false // Quick return.
	case ifst < 0 || ifst >= n:
		panic(badIfst)
	case ilst < 0 || ilst >= n:
		panic(badIlst)
	}
	var (
		here, nbnext int
	)
	nbf := 1
	nbl := 1
	// Determine the first row of the specified block
	// and find out if it is 1×1 or 2×2.
	if ifst > 0 && a[ifst*lda+ifst-1] != 0 {
		ifst--
	}
	if ifst < n-1 && a[(ifst+1)*lda+ifst] != 0 {
		nbf = 2
	}

	// Determine the first row of the final block
	// and find out if it is 1×1 or 2×2.

	if ilst > 0 && a[ilst*lda+ilst-1] != 0 {
		ilst--
	}
	if ilst < n-1 && a[(ilst+1)*lda+ilst] != 0 {
		nbl = 2
	}
	if ifst == ilst {
		return ifst, ilst, false, false
	}
	if ifst < ilst {
		// Update ilst.
		if nbf == 2 && nbl == 1 {
			ilst--
		}
		if nbf == 1 && nbl == 2 {
			ilst++
		}
		here = ifst
	Ten:

		// Swap with next one below.
		if nbf == 1 || nbf == 2 {

			// Current block is either 1×1 or 2×2.

			nbnext = 1
			if (here+nbf+1 <= n) && a[(here+nbf)*lda+here+nbf-1] != 0 {
				nbnext = 2
			}
			ill, badWork := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, nbf, nbnext, work)
			if ill || badWork {
				ilst = here
				return ifst, ilst, ill, badWork
			}
			here += nbnext

			// Test if 2×2 block breaks into two 1×1 blocks.
			if nbf == 2 && a[(here+1)*lda+here] == 0 {
				nbf = 3
			}
		} else {

			// Current block consists of two 1×1 blocks,
			// each of which must be swapped individually.

			nbnext = 1
			if here+3 < n && a[(here+3)*lda+here+2] != 0 {
				nbnext = 2
			}
			ill, badWork := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here+1, 1, nbnext, work)
			if ill || badWork {
				ilst = here
				return ifst, ilst, ill, badWork
			}
			if nbnext == 1 {

				// Swap two 1×1 blocks.

				ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work)
				if ill || badWork {
					ilst = here
					return ifst, ilst, ill, badWork
				}
				here++
			} else {
				// Recompute nbnext in case of 2×2 split.
				if a[(here+2)*lda+here+1] == 0 {
					nbnext = 1
				}
				if nbnext == 2 {

					// 2×2 block did NOT split.

					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, nbnext, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here += 2
				} else {

					// 2×2 block split.

					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here++
					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here++
				}
			}
		}
		if here < ilst {
			goto Ten
		}
	} else { // Main if: ifst > ilst

		here = ifst
	Twenty:

		// Swap with next one below.
		if nbf == 1 || nbf == 2 {
			nbnext = 1
			if here >= 2 && a[(here-1)*lda+here-2] != 0 {
				nbnext = 2
			}
			ill, badWork := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbnext, nbnext, nbf, work)
			if ill || badWork {
				ilst = here
				return ifst, ilst, ill, badWork
			}
			here -= nbnext

			// Test if 2×2 block breaks into two 1×1 blocks.
			if nbf == 2 && a[(here+1)*lda+here] == 0 {
				nbf = 3
			}

		} else {

			// Current block consists of two 1×1 blocks,
			// each of which must be swapped individually.

			nbnext = 1
			if here >= 2 && a[(here-1)*lda+here-2] != 0 {
				nbnext = 2
			}
			ill, badWork := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbnext, nbnext, 1, work)
			if ill || badWork {
				ilst = here
				return ifst, ilst, ill, badWork
			}
			if nbnext == 1 {

				// Swap two 1×1 blocks.

				ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, nbnext, 1, work)
				if ill || badWork {
					ilst = here
					return ifst, ilst, ill, badWork
				}
				here--
			} else {
				// Recompute nbnext in case of 2×2 split.
				if a[here*lda+here-1] == 0 {
					nbnext = 1
				}
				if nbnext == 2 {

					// 2×2 block did NOT split.

					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-1, 2, 1, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here -= 2
				} else {

					// 2×2 block split.

					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here--
					ill, badWork = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work)
					if ill || badWork {
						ilst = here
						return ifst, ilst, ill, badWork
					}
					here--
				}
			}
		}
		if here > ilst {
			goto Twenty
		}
	}
	ilst = here
	work[0] = float64(lwmin)
	return ifst, ilst, false, false
}
