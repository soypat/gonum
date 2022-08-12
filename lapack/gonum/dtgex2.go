// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE dtgex2(     WANTQ,      WANTZ, N     , A         , LDA    , B          , LDB    , Q          , LDQ    , Z          ,     LDZ, J1    , N1    ,    N2,           WORK, LWORK, INFO )
func (impl Implementation) Dtgex2(wantq bool, wantz bool, n int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int, j1 int, n1 int, n2 int, work []float64, lwork int) (info int) {
	bi := blas64.Implementation()
	var ldst int
	var iwork []int
	ipiv := []int{0, 0, 0, 0, 0, 0, 0, 0}
	jpiv := []int{0, 0, 0, 0, 0, 0, 0, 0}

	_ = iwork
	var wands bool
	ldst = 4
	wands = true
	var strong, weak bool
	var i, idum, linfo, m int
	_ = idum
	var bqra21, brqa21, ddum, dnorma, dnormb, dscale, dsum, eps, f, g, sa, sb, scale, smlnum, thresha, threshb float64
	_ = ddum
	//var iwork [ldst]int
	//var ai,ar,be [2]float64
	var ir, ircop, li, licop, s, scpy, t, tcpy []float64 //var ir , ircop, li , licop, s, scpy, t,tcpy [ldst*ldst]float64
	var taul, taur []float64                             //var taul, taur [ldst]float64
	//var   dlamch float64
	ir = append(ir, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	ircop = append(ircop, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	li = append(li, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	licop = append(licop, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	li = append(li, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	s = append(s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	scpy = append(scpy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	t = append(t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	taul = append(taul, 0, 0, 0, 0)
	taur = append(taur, 0, 0, 0, 0)
	tcpy = append(tcpy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
	info = 0
	if n <= 1 || n1 <= 0 || n2 <= 0 {
		return info
	}
	if n1 > n || (j1+n1) > n {
		return info
	}
	m = n1 + n2
	if lwork < max(1, n*m) || lwork < max(n*m, m*m*2) { //IF( lwork.LT.max( 1, n*m, m*m*2 ) ) THEN
		info = -16

		work[0] = float64(max(max(1, n*m), m*m*2)) //work( 1 ) = max( 1, n*m, m*m*2 )
		return info
	}

	weak = false
	strong = false
	impl.Dlaset(blas.All, ldst, ldst, 0, 0, li, ldst)
	impl.Dlaset(blas.All, ldst, ldst, 0, 0, ir, ldst)
	impl.Dlacpy(blas.All, m, m, a[(j1-1+lda*(j1-1)):], lda, s, ldst)
	impl.Dlacpy(blas.All, m, m, b[(j1-1+ldb*(j1-1)):], ldb, t, ldst)
	eps = dlamchP
	smlnum = dlamchS / eps
	dscale = 0
	dsum = 1

	impl.Dlacpy(blas.All, m, m, s, ldst, work, m)

	dscale, dsum = impl.Dlassq(m*m, work, 1, dscale, dsum)
	dnorma = dscale * math.Sqrt(dsum)
	dscale = 0
	dsum = 1
	impl.Dlacpy(blas.All, m, m, t, ldst, work, m)
	dscale, dsum = impl.Dlassq(m*m, work, 1, dscale, dsum)
	dnormb = dscale * math.Sqrt(dsum)
	thresha = math.Max((20 * eps * dnorma), (smlnum))
	threshb = math.Max((20 * eps * dnormb), (smlnum))
	if m == 2 {
		f = s[2-1+ldst*(2-1)]*t[0] - t[1+ldst]*s[0]          // f = s( 2, 2 )*t( 1, 1 ) - t( 2, 2 )*s( 1, 1 )
		g = s[1+ldst]*t[ldst] - t[1+ldst]*s[ldst]            //g = s( 2, 2 )*t( 1, 2 ) - t( 2, 2 )*s( 1, 2 )
		sa = (math.Abs((s[1+ldst])) * math.Abs((t[0])))      // sa = abs( s( 2, 2 ) ) * abs( t( 1, 1 ) )
		sb = (math.Abs((s[0])) * math.Abs((t[1+ldst])))      //sb = abs( s( 1, 1 ) ) * abs( t( 2, 2 ) )
		ir[ldst], ir[0], ddum = impl.Dlartg(f, g)            //CALL dlartg( f, g, ir( 1, 2 ), ir( 1, 1 ), ddum )
		ir[1] = -ir[ldst]                                    //ir( 2, 1 ) = -ir( 1, 2 )
		ir[1+ldst] = ir[0]                                   //ir( 2, 2 ) = ir( 1, 1 )
		bi.Drot(2, s[0:], 1, s[1+ldst:], 1, ir[0], ir[1])    //CALL drot( 2, s( 1, 1 ), 1, s( 1, 2 ), 1, ir( 1, 1 ),ir( 2, 1 ) )
		bi.Drot(2, t[0:], 1, t[ldst:], 1, ir[0], ir[1+ldst]) //CALL drot( 2, t( 1, 1 ), 1, t( 1, 2 ), 1, ir( 1, 1 ),ir( 2, 1 ) )
		if sa >= sb {
			li[0], li[1], ddum = impl.Dlartg(s[0], s[1]) //CALL dlartg( s( 1, 1 ), s( 2, 1 ), li( 1, 1 ), li( 2, 1 ),ddum )
		} else {
			li[0], li[1], ddum = impl.Dlartg(t[0], t[1]) //CALL dlartg( t( 1, 1 ), t( 2, 1 ), li( 1, 1 ), li( 2, 1 ),ddum )
		}
		bi.Drot(2, s[0:], ldst, s[1:], ldst, li[0], li[1])                    //CALL drot( 2, s( 1, 1 ), ldst, s( 2, 1 ), ldst, li( 1, 1 ),li( 2, 1 ) )
		bi.Drot(2, t[0:], ldst, t[1:], ldst, li[0], li[1])                    //CALL drot( 2, t( 1, 1 ), ldst, t( 2, 1 ), ldst, li( 1, 1 ),li( 2, 1 ) )
		li[1+ldst] = li[0]                                                    //li( 2, 2 ) = li( 1, 1 )
		li[ldst] = -li[1]                                                     //li( 1, 2 ) = -li( 2, 1 )
		weak = math.Abs((s[1])) <= (thresha) && math.Abs((t[1])) <= (threshb) //weak = abs( s( 2, 1 ) ) .LE. thresha .AND.abs( t( 2, 1 ) ) .LE. threshb
		if !weak {
			goto G70
		}

		if wands {
			impl.Dlacpy(blas.All, m, m, a[j1-1+lda*(j1-1):], lda, work[m*m:], m) //CALL dlacpy( 'Full', m, m, a( j1, j1 ), lda, work( m*m+1 ),m )
			//bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li, ldst, s, ldst, 0, work, m)      //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, s, ldst, zero,work, m )
			//bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'T', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )
			impl.Dgemm2('N', 'N', m, m, m, 1, li, ldst, s, ldst, 0, work, m)        //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, s, ldst, zero,work, m )
			impl.Dgemm2('N', 'T', m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'T', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )

			dscale = 0
			dsum = 1
			dscale, dsum = impl.Dlassq(m*m, work[m*m:], 1, dscale, dsum) //CALL dlassq( m*m, work( m*m+1 ), 1, dscale, dsum )
			sa = dscale * math.Sqrt(dsum)
			impl.Dlacpy(blas.All, m, m, b[j1-1+ldb*(j1-1):], ldb, work[m*m:], m) //CALL dlacpy( 'Full', m, m, b( j1, j1 ), ldb, work( m*m+1 ),m )
			//bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li, ldst, t, ldst, 0, work, m)      //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, t, ldst, zero,work, m )
			//bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'T', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )
			impl.Dgemm2('N', 'N', m, m, m, 1, li, ldst, t, ldst, 0, work, m)        //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, t, ldst, zero,work, m )
			impl.Dgemm2('N', 'T', m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'T', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )

			dscale = 0
			dsum = 1
			dscale, dsum = impl.Dlassq(m*m, work[m*m:], 1, dscale, dsum) //CALL dlassq( m*m, work( m*m+1 ), 1, dscale, dsum )
			sb = dscale * math.Sqrt(dsum)
			strong = sa <= thresha && sb <= threshb
			if !strong {
				goto G70
			}
		}

		bi.Drot(j1+1, a[lda*(j1-1):], 1, a[lda*j1:], 1, ir[0], ir[1])                   //CALL drot( j1+1, a( 1, j1 ), 1, a( 1, j1+1 ), 1, ir( 1, 1 ),ir( 2, 1 ) )
		bi.Drot(j1+1, b[ldb*(j1-1):], 1, b[ldb*j1:], 1, ir[0], ir[1])                   //CALL drot( j1+1, b( 1, j1 ), 1, b( 1, j1+1 ), 1, ir( 1, 1 ),ir( 2, 1 ) )
		bi.Drot(n-j1+1, a[j1-1+lda*(j1-1):], lda, a[j1+lda*(j1-1):], lda, li[0], li[1]) //CALL drot( n-j1+1, a( j1, j1 ), lda, a( j1+1, j1 ), lda,li( 1, 1 ), li( 2, 1 ) )
		bi.Drot(n-j1+1, b[j1-1+ldb*(j1-1):], ldb, b[j1+ldb*(j1-1):], ldb, li[0], li[1]) //CALL drot( n-j1+1, b( j1, j1 ), ldb, b( j1+1, j1 ), ldb,li( 1, 1 ), li( 2, 1 ) )
		a[j1+lda*(j1-1)] = 0
		b[j1+ldb*(j1-1)] = 0
		if wantz {
			bi.Drot(n, z[ldz*(j1-1):], 1, z[ldz*j1:], 1, ir[0], ir[1]) //CALL drot( n, z( 1, j1 ), 1, z( 1, j1+1 ), 1, ir( 1, 1 ),ir( 2, 1 ) )
		}
		if wantq {
			bi.Drot(n, q[ldq*(j1-1):], 1, q[ldq*j1:], 1, li[0], li[1]) //CALL drot( n, q( 1, j1 ), 1, q( 1, j1+1 ), 1, li( 1, 1 ),li( 2, 1 ) )
		}
		return

	} else {

		impl.Dlacpy(blas.All, n1, n2, t[ldst*n1:], ldst, li, ldst) //CALL dlacpy( 'Full', n1, n2, t( 1, n1+1 ), ldst, li, ldst )
		liaux := make([]float64, 16)
		var ix, jx int
		for ix = 1; ix <= 4; ix++ {
			for jx = 1; jx <= 4; jx++ {
				//zaux[jx][ix] = z[ix][jx]
				liaux[jx-1+ldst*(ix-1)] = li[ix-1+ldst*(jx-1)]
			}
		}
		saux := make([]float64, 16)
		for ix = 1; ix <= 4; ix++ {
			for jx = 1; jx <= 4; jx++ {
				//zaux[jx][ix] = z[ix][jx]
				saux[jx-1+ldst*(ix-1)] = s[ix-1+ldst*(jx-1)]
			}
		}

		//impl.Dlacpy(blas.All, n1, n2, s[ldst*n1:], ldst, ir[n2+ldst*n1:], ldst) //CALL dlacpy( 'Full', n1, n2, s( 1, n1+1 ), ldst,ir( n2+1, n1+1 ), ldst )
		impl.Dlacpy(blas.All, n1, n2, saux[(n1):], ldst, ir[n1+ldst*n2:], ldst) //CALL dlacpy( 'Full', n1, n2, s( 1, n1+1 ), ldst,ir( n2+1, n1+1 ), ldst )
		iraux := make([]float64, 16)
		for ix = 1; ix <= 4; ix++ {
			for jx = 1; jx <= 4; jx++ {
				//zaux[jx][ix] = z[ix][jx]
				iraux[jx-1+ldst*(ix-1)] = ir[ix-1+ldst*(jx-1)]
			}
		}
		var ijob2 lapack.MaximizeNormXJob
		ijob2 = 0
		if len(iwork) < (m + n + 6) {
			var aux int
			for aux = len(iwork); aux+1 <= (n1 + n2 + 6); aux++ {
				iwork = append(iwork, 0)
			}
		}

		//scale, dsum, dscale, idum, linfo = impl.Dtgsy2('N', ijob2, n1, n2, s, ldst, saux[n1+ldst*(n1):], ldst, iraux[(n2)+ldst*(n1):], ldst, t, ldst, t[n1+ldst*(n1):], ldst, liaux, ldst, dsum, scale, iwork, ipiv, jpiv)
		scale, dsum, dscale, idum, linfo = impl.Dtgsy2('N', ijob2, n1, n2, s, ldst, s[n1+ldst*(n1):], ldst, iraux[(n2)+ldst*(n1):], ldst, t, ldst, t[n1+ldst*(n1):], ldst, liaux, ldst, dsum, scale, iwork, ipiv, jpiv)
		//CALL tgsy2( 'N', 0, n1, n2, s, ldst, s( n1+1, n1+1 ), ldst,ir( n2+1, n1+1 ), ldst, t, ldst, t( n1+1, n1+1 ),ldst, li, ldst,
		//scale, dsum, dscale, iwork, idum,linfo )
		li = liaux
		// Dtgsy2(trans blas.Transpose, ijob lapack.MaximizeNormXJob, m, n int, a []float64, lda int,         b []float64, ldb int,      c []float64, ldc int, d []float64, ldd int,         e []float64, lde int, f []float64, ldf int, rdsum, rdscal float64, iwork []int, ipiv []int, jpiv []int)
		//CALL dtgsy2( 'N', 0, n1, n2, s, ldst, s( n1+1, n1+1 ), ldst,ir( n2+1, n1+1 )    , ldst, t, ldst, t( n1+1, n1+1 )     ,ldst, li, ldst, scale, dsum, dscale, iwork, idum,linfo )
		if linfo != 0 {
			goto G70
		}

		//DO 10 i = 1, n2
		for i := 1; i <= n2; i++ {
			bi.Dscal(n1, -1, li[ldst*(i-1):], 1) //CALL dscal( n1, -one, li( 1, i ), 1 )
			li[n1+i-1+ldst*(i-1)] = scale        //li( n1+i, i ) = scale
		} //10    CONTINUE
		//impl.Dgeqr2(n2+1, m, li, ldst, taul, work[12:]) //CALL dgeqr2( m, n2, li, ldst, taul, work, linfo )
		workaux := make([]float64, 16)
		for ix = 1; ix <= 4; ix++ {
			for jx = 1; jx <= 4; jx++ {
				//zaux[jx][ix] = z[ix][jx]
				workaux[jx-1+ldst*(ix-1)] = work[ix-1+ldst*(jx-1)]
			}
		}
		impl.Dgeqr22(m, n2, li, ldst, taul, work)
		if linfo != 0 {
			goto G70
		}
		impl.Dorg2r(m, m, n2, li, ldst, taul, work) //CALL dorg2r( m, m, n2, li, ldst, taul, work, linfo )
		if linfo != 0 {
			goto G70
		}

		//DO 20 i = 1, n1
		for i := 1; i <= n1; i++ {
			ir[n2+i-1+ldst*(i-1)] = scale //ir( n2+i, i ) = scale
		} //20    CONTINUE
		impl.Dgerq2(n1, m, ir[n2:], ldst, taur, work) //CALL dgerq2( n1, m, ir( n2+1, 1 ), ldst, taur, work, linfo )
		if linfo != 0 {
			goto G70
		}
		impl.Dorgr2(m, m, n1, ir, ldst, taur, work) //CALL dorgr2( m, m, n1, ir, ldst, taur, work, linfo )
		if linfo != 0 {
			goto G70
		}

		bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, li, ldst, s, ldst, 0, work, m) //CALL dgemm( 'T', 'N', m, m, m, one, li, ldst, s, ldst, zero,work, m )
		bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, 1, work, m, ir, ldst, 0, s, ldst) //CALL dgemm( 'N', 'T', m, m, m, one, work, m, ir, ldst, zero, s,ldst )
		bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, li, ldst, t, ldst, 0, work, m) //CALL dgemm( 'T', 'N', m, m, m, one, li, ldst, t, ldst, zero,work, m )
		bi.Dgemm(blas.NoTrans, blas.Trans, m, m, m, 1, work, m, ir, ldst, 0, t, ldst) //CALL dgemm( 'N', 'T', m, m, m, one, work, m, ir, ldst, zero, t,ldst )
		impl.Dlacpy(blas.All, m, m, s, ldst, scpy, ldst)                              //CALL dlacpy( 'F', m, m, s, ldst, scpy, ldst )
		impl.Dlacpy(blas.All, m, m, t, ldst, tcpy, ldst)                              //CALL dlacpy( 'F', m, m, t, ldst, tcpy, ldst )
		impl.Dlacpy(blas.All, m, m, ir, ldst, ircop, ldst)                            //CALL dlacpy( 'F', m, m, ir, ldst, ircop, ldst )
		impl.Dlacpy(blas.All, m, m, li, ldst, licop, ldst)                            //CALL dlacpy( 'F', m, m, li, ldst, licop, ldst )
		impl.Dgerq2(m, m, t, ldst, taur, work)                                        //CALL dgerq2( m, m, t, ldst, taur, work, linfo )
		if linfo != 0 {
			goto G70
		}
		impl.Dormr2('R', 'T', m, m, m, t, ldst, taur, s, ldst, work) //CALL dormr2( 'R', 'T', m, m, m, t, ldst, taur, s, ldst, work,linfo )
		if linfo != 0 {
			goto G70
		}
		impl.Dormr2('L', 'N', m, m, m, t, ldst, taur, ir, ldst, work) //CALL dormr2( 'L', 'N', m, m, m, t, ldst, taur, ir, ldst, work,linfo )
		if linfo != 0 {
			goto G70
		}
		dscale = 0
		dsum = 1
		//DO 30 i = 1, n2
		for i = 1; i <= n2; i++ {
			impl.Dlassq(n1, s[n2+ldst*(i-1):], 1, dscale, dsum) //CALL dlassq( n1, s( n2+1, i ), 1, dscale, dsum )
		} //	 30    CONTINUE
		brqa21 = dscale * math.Sqrt(dsum)

		impl.Dgeqr2(m, m, tcpy, ldst, taul, work) //CALL dgeqr2( m, m, tcpy, ldst, taul, work, linfo )
		if linfo != 0 {
			goto G70
		}
		impl.Dorm2r('L', 'T', m, m, m, tcpy, ldst, taul, scpy, ldst, work)  //CALL dorm2r( 'L', 'T', m, m, m, tcpy, ldst, taul, scpy, ldst,work, info )
		impl.Dorm2r('R', 'N', m, m, m, tcpy, ldst, taul, licop, ldst, work) //CALL dorm2r( 'R', 'N', m, m, m, tcpy, ldst, taul, licop, ldst,work, info )
		if linfo != 0 {
			goto G70
		}
		dscale = 0
		dsum = 1
		//DO 40 i = 1, n2
		for i = 1; i <= n2; i++ {
			impl.Dlassq(n1, scpy[n2+ldst*(i-1):], 1, dscale, dsum) //CALL dlassq( n1, scpy( n2+1, i ), 1, dscale, dsum )
		} //40    CONTINUE
		bqra21 = dscale * math.Sqrt(dsum)
		if bqra21 <= brqa21 && bqra21 <= thresha {
			impl.Dlacpy(blas.All, m, m, scpy, ldst, s, ldst)   //CALL dlacpy( 'F', m, m, scpy, ldst, s, ldst )
			impl.Dlacpy(blas.All, m, m, tcpy, ldst, t, ldst)   //CALL dlacpy( 'F', m, m, tcpy, ldst, t, ldst )
			impl.Dlacpy(blas.All, m, m, ircop, ldst, ir, ldst) //CALL dlacpy( 'F', m, m, ircop, ldst, ir, ldst )
			impl.Dlacpy(blas.All, m, m, licop, ldst, li, ldst) //CALL dlacpy( 'F', m, m, licop, ldst, li, ldst )
		} else if brqa21 >= thresha {
			goto G70
		}
		impl.Dlaset(blas.Lower, m-1, m-1, 0, 0, t[1:], ldst) //CALL dlaset( 'Lower', m-1, m-1, zero, zero, t(2,1), ldst )

		if wands {
			impl.Dlacpy(blas.All, m, m, a[j1-1+lda*(j1-1):], lda, work[m*m:], m)                   //CALL dlacpy( 'Full', m, m, a( j1, j1 ), lda, work( m*m+1 ),m )
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li, ldst, s, ldst, 0, work, m)        //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, s, ldst, zero,work, m )
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'N', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )
			dscale = 0
			dsum = 1
			impl.Dlassq(m*m, work[m*m:], 1, dscale, dsum) //CALL dlassq( m*m, work( m*m+1 ), 1, dscale, dsum )
			sa = dscale * math.Sqrt(dsum)

			impl.Dlacpy(blas.All, m, m, b[j1-1+ldb*(j1-1):], ldb, work[m*m:], m)                   //CALL dlacpy( 'Full', m, m, b( j1, j1 ), ldb, work( m*m+1 ),m )
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li, ldst, t, ldst, 0, work, m)        //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, t, ldst, zero,work, m )
			bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, -1, work, m, ir, ldst, 1, work[m*m:], m) //CALL dgemm( 'N', 'N', m, m, m, -one, work, m, ir, ldst, one,work( m*m+1 ), m )
			dscale = 0
			dsum = 1
			impl.Dlassq(m*m, work[m*m:], 1, dscale, dsum) //CALL dlassq( m*m, work( m*m+1 ), 1, dscale, dsum )
			sb = dscale * math.Sqrt(dsum)
			strong = sa <= thresha && sb <= threshb
			if !strong {
				goto G70
			}
		}
		impl.Dlaset(blas.All, n1, n2, 0, 0, s[n2:], ldst)              //CALL dlaset( 'Full', n1, n2, zero, zero, s(n2+1,1), ldst )
		impl.Dlacpy(blas.All, m, m, s, ldst, a[j1-1+lda*(j1-1):], lda) //CALL dlacpy( 'F', m, m, s, ldst, a( j1, j1 ), lda )
		impl.Dlacpy(blas.All, m, m, t, ldst, b[j1-1+ldb*(j1-1):], ldb) //CALL dlacpy( 'F', m, m, t, ldst, b( j1, j1 ), ldb )
		impl.Dlaset(blas.All, ldst, ldst, 0, 0, t, ldst)               //CALL dlaset( 'Full', ldst, ldst, zero, zero, t, ldst )
		impl.Dlaset(blas.All, m, m, 0, 0, work, m)                     //CALL dlaset( 'Full', m, m, zero, zero, work, m )
		work[0] = 1                                                    //work( 1 ) = one
		t[0] = 1                                                       //t( 1, 1 ) = one
		idum = lwork - m*m - 2
		if n2 > 1 {
			//CALL dlagv2( a( j1, j1 ), lda, b( j1, j1 ), ldb, ar, ai, be,work( 1 ), work( 2 ), t( 1, 1 ), t( 2, 1 ) )
			work[m] = -work[1]         //work( m+1 ) = -work( 2 )
			work[m+1] = work[0]        //work( m+2 ) = work( 1 )
			t[n2-1+ldst*(n2-1)] = t[0] //t( n2, n2 ) = t( 1, 1 )
			t[ldst*(2-1)] = -t[1]      //t( 1, 2 ) = -t( 2, 1 )
		}
		work[m*m-1] = 1 //work( m*m ) = one
		t[m*m-1] = 1    //t( m, m ) = one

		if n1 > 1 {
			//CALL dlagv2( a( j1+n2, j1+n2 ), lda, b( j1+n2, j1+n2 ), ldb,taur, taul, work( m*m+1 ), work( n2*m+n2+1 ),work( n2*m+n2+2 ), t( n2+1, n2+1 ),t( m, m-1 ) )
			work[m*m-1] = work[n2*m+n2]         //work( m*m ) = work( n2*m+n2+1 )
			work[m*m] = -work[n2*m+n2+1]        //work( m*m-1 ) = -work( n2*m+n2+2 )
			t[m-1+ldst*(m-1)] = t[n2+ldst*(n2)] //t( m, m ) = t( n2+1, n2+1 )
			t[m+ldst*(m-1)] = -t[m-1+ldst*(m)]  //t( m-1, m ) = -t( m, m-1 )
		}
		bi.Dgemm(blas.Trans, blas.NoTrans, n2, n1, n2, 1, work, m, a[j1-1+lda*(j1+n2-1):], lda, 0, work[m*m:], n2)            //CALL dgemm( 'T', 'N', n2, n1, n2, one, work, m, a( j1, j1+n2 ),lda, zero, work( m*m+1 ), n2 )
		impl.Dlacpy(blas.All, n2, n1, work[m*m:], n2, a[j1-1+lda*(j1*n2-1):], lda)                                            //CALL dlacpy( 'Full', n2, n1, work( m*m+1 ), n2, a( j1, j1+n2 ),lda )
		bi.Dgemm(blas.Trans, blas.NoTrans, n2, n1, n2, 1, work, m, b[j1-1+ldb*(j1+n2-1):], ldb, 0, work[m*m:], n2)            //CALL dgemm( 'T', 'N', n2, n1, n2, one, work, m, b( j1, j1+n2 ),ldb, zero, work( m*m+1 ), n2 )
		impl.Dlacpy(blas.All, n2, n1, work[m*m:], n2, b[j1-1+ldb*(j1+n2-1):], ldb)                                            //CALL dlacpy( 'Full', n2, n1, work( m*m+1 ), n2, b( j1, j1+n2 ),ldb )
		bi.Dgemm(blas.NoTrans, blas.NoTrans, m, m, m, 1, li, ldst, work, m, 0, work[m*m:], m)                                 //CALL dgemm( 'N', 'N', m, m, m, one, li, ldst, work, m, zero,work( m*m+1 ), m )
		impl.Dlacpy(blas.All, m, m, work[m*m:], m, li, ldst)                                                                  //CALL dlacpy( 'Full', m, m, work( m*m+1 ), m, li, ldst )
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n2, n1, n1, 1, a[j1-1+lda*(j1+n2-1):], lda, t[n2+ldst*(n2):], ldst, 0, work, n2) //CALL dgemm( 'N', 'N', n2, n1, n1, one, a( j1, j1+n2 ), lda, t( n2+1, n2+1 ), ldst, zero, work, n2 )
		impl.Dlacpy(blas.All, n2, n1, work, n2, a[j1-1+lda*(j1+n2-1):], lda)                                                  //CALL dlacpy( 'Full', n2, n1, work, n2, a( j1, j1+n2 ), lda )
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n2, n1, n1, 1, b[j1-1+ldb*(j1+n2-1):], ldb, t[n2+ldst*(n2):], ldst, 0, work, n2) //CALL dgemm( 'N', 'N', n2, n1, n1, one, b( j1, j1+n2 ), ldb,t( n2+1, n2+1 ), ldst, zero, work, n2 )
		impl.Dlacpy(blas.All, n2, n1, work, n2, b[j1-1+ldb*(j1+n2-1):], ldb)                                                  //CALL dlacpy( 'Full', n2, n1, work, n2, b( j1, j1+n2 ), ldb )
		bi.Dgemm(blas.Trans, blas.NoTrans, m, m, m, 1, ir, ldst, t, ldst, 0, work, m)                                         //CALL dgemm( 'T', 'N', m, m, m, one, ir, ldst, t, ldst, zero,work, m )
		impl.Dlacpy(blas.All, m, m, work, m, ir, ldst)                                                                        //CALL dlacpy( 'Full', m, m, work, m, ir, ldst )
		if wantq {
			bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, q[ldq*(j1-1):], ldq, li, ldst, 0, work, n) //CALL dgemm( 'N', 'N', n, m, m, one, q( 1, j1 ), ldq, li,ldst, zero, work, n )
			impl.Dlacpy(blas.All, n, m, work, n, q[ldq*(j1-1):], ldq)                                   //CALL dlacpy( 'Full', n, m, work, n, q( 1, j1 ), ldq )
		}
		if wantz {
			bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, z[ldz*(j1-1):], ldz, ir, ldst, 0, work, n) //CALL dgemm( 'N', 'N', n, m, m, one, z( 1, j1 ), ldz, ir,ldst, zero, work, n )
			impl.Dlacpy(blas.All, n, m, work, n, z[ldz*(j1-1):], ldz)                                   //impl.Dlacpy( 'Full', n, m, work, n, z( 1, j1 ), ldz )

		}

		i = j1 + m
		if i <= n {
			bi.Dgemm(blas.Trans, blas.NoTrans, m, n-i+1, m, 1, li, ldst, a[j1-1+lda*(i-1):], lda, 0, work, m) //CALL dgemm( 'T', 'N', m, n-i+1, m, one, li, ldst,a( j1, i ), lda, zero, work, m )
			impl.Dlacpy(blas.All, m, n-i+1, work, m, a[j1-1+lda*(i-1):], lda)                                 //CALL dlacpy( 'Full', m, n-i+1, work, m, a( j1, i ), lda )
			bi.Dgemm(blas.Trans, blas.NoTrans, m, n-i+1, m, 1, li, ldst, b[j1-1+ldb*(i-1):], ldb, 0, work, m) //CALL dgemm( 'T', 'N', m, n-i+1, m, one, li, ldst,b( j1, i ), ldb, zero, work, m )
			impl.Dlacpy(blas.All, m, n-i+1, work, m, b[j1-1+ldb*(i-1):], ldb)                                 //CALL dlacpy( 'Full', m, n-i+1, work, m, b( j1, i ), ldb )
		}
		i = j1 - 1
		if i > 0 {
			bi.Dgemm(blas.NoTrans, blas.NoTrans, i, m, m, 1, a[lda*(j1-1):], lda, ir, ldst, 0, work, i) //CALL dgemm( 'N', 'N', i, m, m, one, a( 1, j1 ), lda, ir,ldst, zero, work, i )
			impl.Dlacpy(blas.All, i, m, work, i, a[lda*(j1-1):], lda)                                   //CALL dlacpy( 'Full', i, m, work, i, a( 1, j1 ), lda )
			bi.Dgemm(blas.NoTrans, blas.NoTrans, i, m, m, 1, b[ldb*(j1-1):], ldb, ir, ldst, 0, work, i) //CALL dgemm( 'N', 'N', i, m, m, one, b( 1, j1 ), ldb, ir,ldst, zero, work, i )
			impl.Dlacpy(blas.All, i, m, work, i, b[ldb*(j1-1):], ldb)                                   //CALL dlacpy( 'Full', i, m, work, i, b( 1, j1 ), ldb )
		}

		return

	}

G70: //70 CONTINUE

	info = 1
	return

}
