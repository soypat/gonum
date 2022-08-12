// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

//import "gonum.org/v1/gonum/blas"

func (impl Implementation) Dtgexc(wantq bool, wantz bool, n int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int, ifstin int, ilst int, work []float64, lwork int) (info int, ifstout int, ilstout int) {

	var lquery bool
	var here, lwmin, nbf, nbl, nbnext int
	info = 0
	lquery = (lwork == -1)
	if n < 0 {
		info = -3
	} else if lda < max(1, n) {
		info = -5
	} else if ldb < max(1, n) {
		info = -7
	} else if (ldq < 1) || wantq && (ldq < max(1, n)) {
		info = -9
	} else if ldz < 1 || wantz && (ldz < max(1, n)) {
		info = -11
	} else if ifstin < 1 || ifstin > n {
		info = -12
	} else if ilst < 1 || ilst > n {
		info = -13
	}

	if info == 0 {
		if n <= 1 {
			lwmin = 1
		} else {
			lwmin = 4*n + 16
		}
		//work[1] = lwmin
		if len(work) < 1 {
			work = append(work, 0)
			work[0] = float64(lwmin)
		} else {
			work[0] = float64(lwmin)
		}

		if lwork < lwmin && !(lquery) {
			info = -15
		}
	}

	if info != 0 {
		//CALL xerbla( 'DTGEXC', -info )
		panic(info)
	} else if lquery {
		return
	}
	if n <= 1 {
		return
	}
	if ifstin > 1 {
		if a[ifstin-1+lda*(ifstin-2)] != 0 { //if( a( ifst, ifst-1 )!=0 ){
			ifstin = ifstin - 1
		}
	}
	nbf = 1
	if ifstin < n {
		if a[ifstin+lda*(ifstin-1)] != 0 { //if( a( ifst+1, ifst )!=0 ){
			nbf = 2
		}
	}

	if ilst > 1 {
		if a[ilst-1+lda*(ilst-2)] != 0 { //if( a( ilst, ilst-1 )!=0 ){
			ilst = ilst - 1
		}
	}
	nbl = 1
	if ilst < n {
		if a[ilst+lda*(ilst-1)] != 0 { //if( a( ilst+1, ilst )!=0 ){
			nbl = 2
		}
	}
	if ifstin == ilst {
		return
	}
	if ifstin < ilst {
		if nbf == 2 && nbl == 1 {
			ilst = ilst - 1
		}
		if nbf == 1 && nbl == 2 {
			ilst = ilst + 1
		}
		here = ifstin

	G10: //CONTINUE
		if nbf == 1 || nbf == 2 {
			nbnext = 1
			if (here + nbf + 1) <= n {
				if a[here+nbf+lda*(here+nbf-1)] != 0 { //if( a( here+nbf+1, here+nbf )!=0 ){
					nbnext = 2
				}
			}
			//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z,ldz, here, nbf, nbnext, work, lwork, info )
			if info != 0 {
				ilst = here
				return info, ifstin, ilst
			}
			here = here + nbnext
			if nbf == 2 {
				if a[here+lda*(here-1)] == 0 { //if( a( here+1, here )==0 ){
					nbf = 3
				}
			}
		} else {
			nbnext = 1
			if (here + 3) <= n {
				if a[here+2+lda*(here+1)] != 0 { //if( a( here+3, here+2 )!=0 ){
					nbnext = 2
				}
			}
			//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z,ldz, here+1, 1, nbnext, work, lwork, info )
			if info != 0 {
				ilst = here
				return info, ifstin, ilst
			}
			if nbnext == 1 {
				//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z,ldz, here, 1, 1, work, lwork, info )
				if info != 0 {
					ilst = here
					return info, ifstin, ilst
				}
				here = here + 1

			} else {
				if a[here+1+lda*(here)] == 0 { //if( a( here+2, here+1 )==0 ){
					nbnext = 1
				}
				if nbnext == 2 {
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq,z, ldz, here, 1, nbnext, work, lwork,info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here + 2
				} else {
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq,z, ldz, here, 1, 1, work, lwork, info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here + 1
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq,z, ldz, here, 1, 1, work, lwork, info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here + 1
				}

			}
		}
		if here < ilst {
			goto G10
		}
	} else {
		here = ifstin

	G20: //CONTINUE
		if nbf == 1 || nbf == 2 {
			nbnext = 1
			if here >= 3 {
				if a[here-2+lda*(here-3)] != 0 { //if( a( here-1, here-2 )!=0 ){
					nbnext = 2
				}
			}
			info = impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbnext, nbnext, nbf, work, lwork)
			nbnext = nbnext
			if info != 0 {
				ilst = here
				return info, ifstin, ilst
			}
			here = here - nbnext
			if nbf == 2 {
				if a[here+lda*(here-1)] == 0 { //if( a( here+1, here )==0 ){
					nbf = 3
				}
			}
		} else {
			nbnext = 1
			if here >= 3 {
				if a[here-2+lda*(here-3)] != 0 { //if( a( here-1, here-2 )!=0 ){
					nbnext = 2
				}
			}
			//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z,ldz, here-nbnext, nbnext, 1, work, lwork,info )
			if info != 0 {
				ilst = here
				return info, ifstin, ilst
			}
			if nbnext == 1 {
				//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z,ldz, here, nbnext, 1, work, lwork, info )
				if info != 0 {
					ilst = here
					return info, ifstin, ilst
				}
				here = here - 1
			} else {
				if a[here-1+lda*(here-2)] == 0 { //if( a( here, here-1 )==0 ){
					nbnext = 1
				}
				if nbnext == 2 {
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq,z, ldz, here-1, 2, 1, work, lwork, info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here - 2
				} else {
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, 1, 1, work, lwork, info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here - 1
					//CALL dtgex2( wantq, wantz, n, a, lda, b, ldb, q, ldq,z, ldz, here, 1, 1, work, lwork, info )
					if info != 0 {
						ilst = here
						return info, ifstin, ilst
					}
					here = here - 1
				}
			}
		}

		if here > ilst {
			goto G20
		}
	}

	ilst = here
	//work[1] =float64( lwmin)
	work[0] = float64(lwmin)
	return info, ifstin, ilst
}
