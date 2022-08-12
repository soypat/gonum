// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

//import "gonum.org/v1/gonum/blas"
//SUBROUTINE              dhgeqz(      JOB,      COMPQ,      COMPZ,     N,     ILO,     IHI,           H,     LDH,           T,     LDT,      ALPHAR, ALPHAI, BETA, Q, LDQ, Z, LDZ, WORK,LWORK, INFO )
func (impl Implementation) Dhgeqz(job byte, compq byte, compz byte, n int, ilo int, ihi int, h []float64, ldh int, t []float64, ldt int, q []float64, ldq int, z []float64, ldz int, lwork int) (alphar []float64, alphai []float64, beta []float64, work []float64, info int) {
	bi := blas64.Implementation()

	//var COMPQ, COMPZ, JOB byte//CHARACTER          COMPQ, COMPZ, JOB
	//var ihi, ilo, info, ldh, ldq, ldt, ldz, lwork, n int            //IHI, ILO, INFO, LDH, LDQ, LDT, LDZ, LWORK, N
	//DOUBLE PRECISION   ALPHAI( * ), ALPHAR( * ), BETA( * ),H( LDH, * ), Q( LDQ, * ), T( LDT, * ),work( * ), z( ldz, * )

	//DOUBLE PRECISION   HALF, ZERO, ONE, SAFETY
	//PARAMETER          ( HALF = 0.5d+0, zero = 0.0d+0, one = 1.0d+0,                 safety = 1.0d+2 )

	var ilazr2, ilazro, ilpivt, ilq, ilschr, ilz, lquery bool
	var icompq, icompz, ifirst, ifrstm, iiter, ilast, ilastm, in, ischur, istart, j, jc, jch, jiter, jr, maxit int //INTEGER            ICOMPQ, ICOMPZ, IFIRST, IFRSTM, IITER, ILAST,ILASTM, IN, ISCHUR, ISTART, J, JC, JCH, JITER,jr, maxit
	var a11, a12, a1i, a1r, a21, a22, a2i, a2r, ad11, ad11l, ad12, ad12l, ad21, ad21l, ad22, ad22l, ad32l, an, anorm, ascale, atol, b11, b1a, b1i, b1r, b22, b2a, b2i, b2r, bn, bnorm, bscale, btol, c, c11i, c11r, c12, c21, c22i, c22r, cl, cq, cr, cz, eshift, s, s1, s1inv, s2, safmax, safmin, scale, sl, sqi, sqr, sr, szi, szr, t1, tau, temp, temp2, tempi, tempr, u1, u12, u12l, u2, ulp, vs, w11, w12, w21, w22, wabs, wi, wr, wr2 float64
	_ = b1i
	_ = b1r
	_ = b2r
	_ = b2i
	_ = b11
	_ = b22
	_ = cl
	_ = cr
	_ = sl
	_ = sr
	_ = temp
	_ = wr2
	_ = s2
	_ = tempr
	var v [3]float64

	var aux int
	for aux = 1; aux <= n; aux++ {
		alphar = append(alphar, 0)
		alphai = append(alphai, 0)
		beta = append(beta, 0)
	}

	//LOGICAL            LSAME
	// DOUBLE PRECISION   DLAMCH, DLANHS, DLAPY2, DLAPY3
	// EXTERNAL           lsame, dlamch, dlanhs, dlapy2, dlapy3
	if job == 'E' {
		ilschr = false
		ischur = 1
	} else if job == 'S' {
		ilschr = true
		ischur = 2
	} else {
		ischur = 0
	}

	if compq == 'N' {
		ilq = false
		icompq = 1
	} else if compq == 'V' {
		ilq = true
		icompq = 2
	} else if compq == 'I' {
		ilq = true
		icompq = 3
	} else {
		icompq = 0
	}

	if compz == 'N' {
		ilz = false
		icompz = 1
	} else if compz == 'V' {
		ilz = true
		icompz = 2
	} else if compz == 'I' {
		ilz = true
		icompz = 3
	} else {
		icompz = 0
	}
	info = 0
	//work[0] = float64(max(1, n)) //work( 1 ) = max( 1, n )
	work = append(work, float64(max(1, n)))
	lquery = (lwork == -1)
	if ischur == 0 {
		info = -1
	} else if icompq == 0 {
		info = -2
	} else if icompz == 0 {
		info = -3
	} else if n < 0 {
		info = -4
	} else if ilo < 1 {
		info = -5
	} else if ihi > n || ihi < ilo-1 {
		info = -6
	} else if ldh < n {
		info = -8
	} else if ldt < n {
		info = -10
	} else if ldq < 1 || (ilq && ldq < n) {
		info = -15
	} else if ldz < 1 || (ilz && ldz < n) {
		info = -17
	} else if lwork < max(1, n) && !lquery {
		info = -19
	}
	if info != 0 {
		//CALL xerbla( 'DHGEQZ', -info )
		panic(info)
	} else if lquery {
		return
	}

	if n <= 0 {
		work[0] = float64(1) //work( 1 ) = dble( 1 )
		return
	}
	if icompq == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if icompz == 3 {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz) //   CALL dlaset( 'Full', n, n, zero, one, z, ldz )
	}
	in = ihi + 1 - ilo
	safmin = dlamchS
	safmax = 1 / safmin
	ulp = dlamchE * dlamchB
	anorm = impl.Dlanhs('F', in, h[ilo-1+ldh*(ilo-1):], ldh, work) //Dlanhs('F', in, h(ilo, ilo), ldh, work)
	bnorm = impl.Dlanhs('F', in, t[ilo-1+ldt*(ilo-1):], ldt, work) //dlanhs('F', in, t(ilo, ilo), ldt, work)
	atol = (math.Max(safmin, (ulp * anorm)))
	btol = (math.Max((safmin), (ulp * bnorm)))
	ascale = (1 / math.Max((safmin), (anorm)))
	bscale = (1 / math.Max((safmin), (bnorm)))
	for j = ihi + 1; j <= n; n++ { //	 DO 30 j = ihi + 1, n
		if t[j-1+ldt*(j-1)] < 0 { //IF( t( j, j ) < zero ) THEN
			if ilschr {
				for jr = 1; jr <= j; jr++ { //DO 10 jr = 1, j
					h[jr-1+ldh*(j-1)] = -h[jr-1+ldh*(j-1)] //h( jr, j ) = -h( jr, j )
					t[jr-1+ldt*(-1)] = -t[jr-1+ldt*(j-1)]  //t( jr, j ) = -t( jr, j )
				} //10          CONTINUE
			} else {
				h[j-1+ldh*(j-1)] = -h[j-1+ldh*(j-1)] //h( j, j ) = -h( j, j )
				t[j-1+ldt*(j-1)] = -t[j-1+ldt*(j-1)] //t( j, j ) = -t( j, j )
			}
			if ilz {
				for jr = 1; jr <= n; jr++ { //DO 20 jr = 1, n
					z[jr-1+ldz*(j-1)] = -z[jr-1+ldz*(j-1)] //z( jr, j ) = -z( jr, j )
				} //20          CONTINUE
			}
		}
		alphar[j-1] = h[j-1+ldh*(j-1)] //alphar( j ) = h( j, j )
		alphai[j-1] = 0                //alphai( j ) = zero
		beta[j-1] = t[j-1+ldt*(j-1)]   //beta( j ) = t( j, j )
	} //30 CONTINUE
	if ihi < ilo {
		goto G380
	}

	ilast = ihi
	if ilschr {
		ifrstm = 1
		ilastm = n
	} else {
		ifrstm = ilo
		ilastm = ihi
	}
	iiter = 0
	eshift = 0
	maxit = 30 * (ihi - ilo + 1)
	//acaaaa
	jiter = 1
G350:
	if jiter <= maxit { //for jiter = 1; jiter <= maxit; jiter++ { //DO 360 jiter = 1, maxit

		if ilast == ilo {

			goto G80
		} else {
			if math.Abs((h[ilast-1+ldh*(ilast-2)])) <= math.Max((safmin), ((ulp)*(math.Abs((h[ilast-1+ldh*(ilast-1)]))+math.Abs((h[ilast-2+ldh*(ilast-2)]))))) {
				h[ilast-1+ldh*(ilast-2)] = 0 //h( ilast, ilast-1 ) = 0
				goto G80
			}
		}

		if math.Abs((t[ilast-1+ldt*(ilast-1)])) <= math.Max((safmin), (ulp)*(math.Abs((t[ilast-2+ldt*(ilast-2)]))+math.Abs((t[ilast-2+ldt*(ilast-2)])))) {
			t[ilast-1+ldt*(ilast-1)] = 0 //t( ilast, ilast ) = 0
			goto G70
		}

		for j = ilast - 1; j >= ilo; j-- { //DO 60 j = ilast - 1, ilo, -1

			if j == ilo {
				ilazro = true
			} else {
				if math.Abs((h[j-1+ldh*(j-2)])) <= math.Max((safmin), (ulp)*(math.Abs((h[j-1+ldh*(j-1)]))+math.Abs((h[j-2+ldh*(j-2)])))) {
					h[j-1+ldh*(j-2)] = 0 //h( j, j-1 ) = 0
					ilazro = true
				} else {
					ilazro = false
				}
			}

			temp = math.Abs((t[j-1+ldt*(j)])) //temp = math.Abs( t( j, j + 1 ) )
			if j > ilo {
				temp = temp + float64(math.Abs((t[j-2+ldt*(j-1)]))) //temp = temp + math.Abs( t( j - 1, j ) )
			}
			if math.Abs((t[j-1+ldt*(j-1)])) < math.Max((safmin), (ulp)*(temp)) { //if( math.Abs( t( j, j ) ) < math.Max( safmath.Min,ulp*temp ) ) THEN
				t[j-1+ldt*(j-1)] = 0 //t( j, j ) = zero

				ilazr2 = false
				if !ilazro {
					temp = (math.Abs((h[j-1+ldh*(j-2)])))  //temp = math.Abs( h( j, j-1 ) )
					temp2 = (math.Abs((h[j-1+ldh*(j-1)]))) //temp2 = math.Abs( h( j, j ) )
					tempr = (math.Max((temp), (temp2)))
					if tempr < 1 && tempr != 0 {
						temp = temp / tempr
						temp2 = temp2 / tempr
					}
					if (temp)*(ascale)*math.Abs((h[j+ldh*(j-1)])) <= (temp2)*(ascale*atol) { //IF( temp*( ascale*math.Abs( h( j+1, j ) ) ) <= temp2*( ascale*atol ) )
						ilazr2 = true
					}
				}

				if ilazro || ilazr2 {
					for jch = j; jch <= (ilast - 1); jch++ { //DO 40 jch = j, ilast - 1
						temp = h[jch-1+ldh*(jch-1)]                                                 //temp = h(jch, jch)
						c, s, h[jch-1+ldh*(jch-1)] = impl.Dlartg2(temp, h[jch+ldh*(jch-1)])         //impl.Dlartg(temp, h[jch+ldh*(jch-1)], c, s, h[jch-1+ldh*(jch-1)])         //CALL dlartg( temp, h( jch+1, jch ), c, s,h( jch, jch ) )
						h[jch+ldh*(jch-1)] = 0                                                      //h( jch+1, jch ) = 0
						bi.Drot(ilastm-jch, h[jch-1+ldh*(jch):], ldh, h[jch+ldh*(jch):], ldh, c, s) //CALL drot( ilastm-jch, h( jch, jch+1 ), ldh,h( jch+1, jch+1 ), ldh, c, s )
						bi.Drot(ilastm-jch, t[jch-1+ldt*(jch):], ldt, t[jch+ldt*(jch):], ldt, c, s) //CALL drot( ilastm-jch, t( jch, jch+1 ), ldt,t( jch+1, jch+1 ), ldt, c, s )
						if ilq {
							bi.Drot(n, q[ldq*(jch-1):], 1, q[ldq*(jch):], 1, c, s) //bi.Drot( n, q( 1, jch ), 1, q( 1, jch+1 ), 1,c, s )
						}
						if ilazr2 {
							h[jch-1+ldh*(jch)] = h[jch-1+ldh*(jch-2)] * c //h( jch, jch-1 ) = h( jch, jch-1 )*c
						}
						ilazr2 = false
						if math.Abs((t[jch+ldt*(jch)])) >= (btol) { //IF( math.Abs( t( jch+1, jch+1 ) ) >= btol ) THEN
							if jch+1 >= ilast {
								goto G80
							} else {
								ifirst = jch + 1
								goto G110
							}
						}
						t[jch+ldt*(jch)] = 0 //t( jch+1, jch+1 ) = zero
					} //40             CONTINUE
					goto G70
				} else {

					for jch = j; jch <= ilast-1; jch++ { //DO 50 jch = j, ilast - 1
						temp = t[jch-1+ldt*(jch)] //temp = t( jch, jch+1 )
						//impl.Dlartg(temp, t[jch+ldt*(jch)], c, s, t[jch-1+ldt*(jch)]) //CALL dlartg( temp, t( jch+1, jch+1 ), c, s,t( jch, jch+1 ) )
						c, s, t[jch-1+ldt*(jch)] = impl.Dlartg2(temp, t[jch+ldt*(jch)])
						t[jch+ldt*(jch)] = 0 //t( jch+1, jch+1 ) = 0
						if jch < ilastm-1 {
							bi.Drot(ilastm-jch-1, t[jch-1+ldt*(jch+1):], ldt, t[jch+ldt*(jch+1):], ldt, c, s) //CALL drot( ilastm-jch-1, t( jch, jch+2 ), ldt,t( jch+1, jch+2 ), ldt, c, s )
						}
						bi.Drot(ilastm-jch+2, h[jch-1+ldh*(jch-2):], ldh, h[jch+ldh*(jch-2):], ldh, c, s) //CALL drot( ilastm-jch+2, h( jch, jch-1 ), ldh,h( jch+1, jch-1 ), ldh, c, s )
						if ilq {
							bi.Drot(n, q[ldq*(jch-1):], 1, q[ldq*(jch):], 1, c, s) //CALL drot( n, q( 1, jch ), 1, q( 1, jch+1 ), 1,c, s )
						}
						temp = h[jch+ldh*(jch-1)]                                                             //temp = h( jch+1, jch )
						c, s, h[jch+ldh*(jch-1)] = impl.Dlartg2(temp, h[jch+ldh*(jch-2)])                     //CALL dlartg( temp, h( jch+1, jch-1 ), c, s,h( jch+1, jch ) )
						h[jch+ldh*(jch-2)] = 0                                                                //h( jch+1, jch-1 ) = zero
						bi.Drot(jch+1-ifrstm, h[ifrstm-1+ldh*(jch-1):], 1, h[ifrstm-1+ldh*(jch-2):], 1, c, s) //CALL drot( jch+1-ifrstm, h( ifrstm, jch ), 1,h( ifrstm, jch-1 ), 1, c, s )
						bi.Drot(jch-ifrstm, t[ifrstm-1+ldt*(jch-1):], 1, t[ifrstm-1+ldt*(jch-2):], 1, c, s)   //            CALL drot( jch-ifrstm, t( ifrstm, jch ), 1,t( ifrstm, jch-1 ), 1, c, s )
						if ilz {
							bi.Drot(n, z[ldz*(jch-1):], 1, z[ldz*(jch-2):], 1, c, s) //CALL drot( n, z( 1, jch ), 1, z( 1, jch-1 ), 1,c, s )
						}
					} //50             CONTINUE
					goto G70
				}
			} else if ilazro {

				ifirst = j
				goto G110
			}

		} //60    CONTINUE

		info = n + 1
		goto G420
	} //  70    CONTINUE
G70:
	temp = h[ilast-1+ldh*(ilast-1)]                                                           //temp = h( ilast, ilast )
	c, s, h[ilast-1+ldh*(ilast-1)] = impl.Dlartg2(temp, h[ilast-1+ldh*(ilast-2)])             //CALL dlartg( temp, h( ilast, ilast-1 ), c, s,h( ilast, ilast ) )
	h[ilast-1+ldh*(ilast-2)] = 0                                                              //h( ilast, ilast-1 ) = zero
	bi.Drot(ilast-ifrstm, h[ifrstm-1+ldh*(ilast-1):], 1, h[ifrstm-1+ldh*(ilast-2):], 1, c, s) //CALL drot( ilast-ifrstm, h( ifrstm, ilast ), 1,h( ifrstm, ilast-1 ), 1, c, s )
	bi.Drot(ilast-ifrstm, t[ifrstm-1+ldt*(ilast-1):], 1, t[ifrstm-1+ldh*(ilast-2):], 1, c, s) //CALL drot( ilast-ifrstm, t( ifrstm, ilast ), 1,t( ifrstm, ilast-1 ), 1, c, s )
	if ilz {
		bi.Drot(n, z[ldz*(ilast-1):], 1, z[ldz*(ilast-2):], 1, c, s) //CALL drot( n, z( 1, ilast ), 1, z( 1, ilast-1 ), 1, c, s )
	}
G80:
	//80    CONTINUE
	if t[ilast-1+ldt*(ilast-1)] < 0 { //if( t( ilast, ilast ) < zero ){
		if ilschr {
			for j = ifrstm; j <= ilast; j++ { //DO 90 j = ifrstm, ilast
				h[j-1+ldh*(ilast-1)] = -h[j-1+ldh*(ilast-1)] //h( j, ilast ) = -h( j, ilast )
				t[j-1+ldt*(ilast-1)] = -t[j-1+ldt*(ilast-1)] //t( j, ilast ) = -t( j, ilast )
			} //90          CONTINUE
		} else {
			h[ilast-1+ldh*(ilast-1)] = -h[ilast-1+ldh*(ilast-1)] //h( ilast, ilast ) = -h( ilast, ilast )
			t[ilast-1+ldt*(ilast-1)] = -t[ilast-1+ldt*(ilast-1)] //t( ilast, ilast ) = -t( ilast, ilast )
		}
		if ilz {
			for j = 1; j <= n; j++ { //DO 100 j = 1, n
				z[j-1+ldz*(ilast-1)] = -z[j-1+ldz*(ilast-1)] //z( j, ilast ) = -z( j, ilast )
			} //100          CONTINUE
		}
	} //de mas?
	alphar[ilast-1] = h[ilast-1+ldh*(ilast-1)] //alphar( ilast ) = h( ilast, ilast ) controlar desde la segunda pasada
	alphai[ilast-1] = 0                        //alphai( ilast ) = zero
	beta[ilast-1] = t[ilast-1+ldt*(ilast-1)]   //beta( ilast ) = t( ilast, ilast )

	ilast = ilast - 1
	if ilast < ilo {
		goto G380
	}

	iiter = 0
	eshift = 0
	if !ilschr {
		ilastm = ilast
		if ifrstm > ilast {
			ifrstm = ilo
		}
	}
	jiter = jiter + 1
	goto G350
G110:
	//}// 110    CONTINUE
	iiter = iiter + 1
	if !ilschr {
		ifrstm = ifirst
	}

	if (iiter/10)*10 == iiter {

		if (float64(maxit)*safmin)*math.Abs(h[ilast+ldh*(ilast-2)]) < math.Abs(t[ilast-2+ldt*(ilast-2)]) { //IF( ( dble( math.Maxit )*safmath.Min )*math.Abs( h( ilast, ilast-1 ) ) < math.Abs( t( ilast-1, ilast-1 ) ) ) THEN
			eshift = h[ilast-1+ldh*(ilast-2)] / t[ilast-2+ldt*(ilast-2)] //eshift = h( ilast, ilast-1 ) /t( ilast-1, ilast-1 )
		} else {
			eshift = eshift + 1/(safmin*float64(maxit)) //eshift = eshift + one / ( safmath.Min*dble( math.Maxit ) )
		}
		s1 = 1
		wr = eshift

	} else {

		//CALL dlag2( h( ilast-1, ilast-1 ), ldh,t( ilast-1, ilast-1 ), ldt, safmath.Min*safety, s1,s2, wr, wr2, wi )
		var i, j int
		var zaux, zaux2 [25]float64
		for i = 1; i <= ldh; i++ {
			for j = 1; j <= ldh; j++ {
				zaux[i-1+ldh*(j-1)] = h[j-1+ldh*(i-1)]
				zaux2[i-1+ldh*(j-1)] = t[j-1+ldh*(i-1)]
			}
		}
		//s1, s2, wr, wr2, wi = impl.Dlag2(h[ilast-2+ldh*(ilast-2):], ldh, t[ilast-2+ldt*(ilast-2):], ldt)
		s1, s2, wr, wr2, wi = impl.Dlag2(zaux[ilast-2+ldh*(ilast-2):], ldh, zaux2[ilast-2+ldt*(ilast-2):], ldt)

		if math.Abs((wr/s1)*t[ilast-1+ldt*(ilast-1)]-h[ilast-1+ldh*(ilast-1)]) > math.Abs((wr2/s2)*t[ilast-1+ldt*(ilast-1)]-h[ilast-1+ldh*(ilast-1)]) { //IF ( math.Abs( (wr/s1)*t( ilast, ilast ) - h( ilast, ilast ) )>  math.Abs( (wr2/s2)*t( ilast, ilast )- h( ilast, ilast ) ) ) THEN
			temp = wr
			wr = wr2
			wr2 = temp
			temp = s1
			s1 = s2
			s2 = temp
		}
		temp = math.Max(s1, safmin*math.Max(1, math.Max(math.Abs(wr), math.Abs(wi))))
		if wi != 0 {
			goto G200
		}
	}

	temp = math.Min(ascale, 1) * (0.5 * safmax)
	if s1 > temp {
		scale = temp / s1
	} else {
		scale = 1
	}

	temp = math.Min(bscale, 1) * (0.5 * safmax)
	if math.Abs(wr) > temp {
		scale = math.Min(scale, temp/math.Abs(wr))
	}
	s1 = scale * s1
	wr = scale * wr

	for j = ilast - 1; j >= ifirst+1; j-- { //DO 120 j = ilast - 1, ifirst + 1, -1
		istart = j
		temp = math.Abs(s1 * h[j-1+ldh*(j-2)])                      //temp = math.Abs( s1*h( j, j-1 ) )
		temp2 = math.Abs(s1*h[j-1+ldh*(j-1)] - wr*t[j-1+ldt*(j-1)]) //temp2 = math.Abs( s1*h( j, j )-wr*t( j, j ) )
		tempr = math.Max(temp, temp2)
		if tempr < 1 && tempr != 0 {
			temp = temp / tempr
			temp2 = temp2 / tempr
		}
		if math.Abs((ascale*h[j+ldh*(j-1)])*temp) <= (ascale*atol)*temp2 { //if( math.Abs( ( ascale*h( j+1, j ) )*temp ) <= ( ascale*atol )*temp2 )GO TO 130
			goto G130
		}
	} //120    CONTINUE

	istart = ifirst
G130:
	//}//130    CONTINUE

	temp = s1*h[istart-1+ldh*(istart-1)] - wr*t[istart-1+ldt*(istart-1)] //temp = s1*h( istart, istart ) - wr*t( istart, istart )
	temp2 = s1 * h[istart+ldh*(istart-1)]   //temp2 = s1*h( istart+1, istart )
	c, s, tempr = impl.Dlartg2(temp, temp2) // CALL dlartg( temp, temp2, c, s, tempr )

	for j = istart; j <= ilast-1; j++ { //DO 190 j = istart, ilast - 1
		if j > istart {
			temp = h[j-1+ldh*(j-2)]                                     //temp = h( j, j-1 )
			c, s, h[j-1+ldh*(j-2)] = impl.Dlartg2(temp, h[j+ldh*(j-2)]) //CALL dlartg( temp, h( j+1, j-1 ), c, s, h( j, j-1 ) )
			h[j+ldh*(j-2)] = 0                                          //h( j+1, j-1 ) = zero
		}

		for jc = j; jc <= ilastm; jc++ { //DO 140 jc = j, ilastm
			temp = c*h[j-1+ldh*(jc-1)] + s*h[j+ldh*(jc-1)]             //temp = c*h( j, jc ) + s*h( j+1, jc )
			h[j+ldh*(jc-1)] = -s*h[j-1+ldh*(jc-1)] + c*h[j+ldh*(jc-1)] //h( j+1, jc ) = -s*h( j, jc ) + c*h( j+1, jc )
			h[j-1+ldh*(jc-1)] = temp                                   //h( j, jc ) = temp
			temp2 = c*t[j-1+ldt*(jc-1)] + s*t[j+ldt*(jc-1)]            //temp2 = c*t( j, jc ) + s*t( j+1, jc )
			t[j+ldt*(jc-1)] = -s*t[j-1+ldt*(jc-1)] + c*t[j+ldt*(jc-1)] //t( j+1, jc ) = -s*t( j, jc ) + c*t( j+1, jc )
			t[j-1+ldt*(jc-1)] = temp2                                  //t( j, jc ) = temp2
		} //140       CONTINUE
		if ilq {
			for jr = 1; jr <= n; jr++ { //DO 150 jr = 1, n
				temp = c*q[jr-1+ldq*(j-1)] + s*q[jr-1+ldq*(j)]             //temp = c*q( jr, j ) + s*q( jr, j+1 )
				q[jr-1+ldq*(j)] = -s*q[jr-1+ldq*(j-1)] + c*q[jr-1+ldq*(j)] //q( jr, j+1 ) = -s*q( jr, j ) + c*q( jr, j+1 )
				q[jr-1+ldq*(j-1)] = temp                                   //q( jr, j ) = temp
			} //150          CONTINUE
		}

		temp = t[j+ldt*(j)] //temp = t( j+1, j+1 )
		//c, s, tempr = impl.Dlartg(temp, temp2) // CALL dlartg( temp, temp2, c, s, tempr )
		//temp, c, s = impl.Dlartg(t[j+ldt*(j-1)], t[j+ldt*(j)]) //CALL dlartg( temp, t( j+1, j ), c, s, t( j+1, j+1 ) )
		c, s, t[j+ldt*(j)] = impl.Dlartg2(temp, t[j+ldt*(j-1)])
		t[j+ldt*(j-1)] = 0 //t( j+1, j ) = zero

		for jr = ifrstm; jr <= min(j+2, ilast); jr++ { //DO 160 jr = ifrstm, math.Min( j+2, ilast )
			temp = c*h[jr-1+ldh*(j)] + s*h[jr-1+ldh*(j-1)]               //temp = c*h( jr, j+1 ) + s*h( jr, j )
			h[jr-1+ldh*(j-1)] = -s*h[jr-1+ldh*(j)] + c*h[jr-1+ldh*(j-1)] //h( jr, j ) = -s*h( jr, j+1 ) + c*h( jr, j )
			h[jr-1+ldh*(j)] = temp                                       //h( jr, j+1 ) = temp
		} //160       CONTINUE
		for jr = ifrstm; jr <= j; jr++ { //DO 170 jr = ifrstm, j
			temp = c*t[jr-1+ldt*(j)] + s*t[jr-1+ldt*(j-1)]               //temp = c*t( jr, j+1 ) + s*t( jr, j )
			t[jr-1+ldt*(j-1)] = -s*t[jr-1+ldt*(j)] + c*t[jr-1+ldt*(j-1)] //t( jr, j ) = -s*t( jr, j+1 ) + c*t( jr, j )
			t[jr-1+ldt*(j)] = temp                                       //t( jr, j+1 ) = temp
		} //170       CONTINUE
		if ilz {
			for jr = 1; jr <= n; jr++ { //DO 180 jr = 1, n
				temp = c*z[jr-1+ldz*(j)] + s*z[jr-1+ldz*(j-1)]               //temp = c*z( jr, j+1 ) + s*z( jr, j )
				z[jr-1+ldz*(j-1)] = -s*z[jr-1+ldz*(j)] + c*z[jr-1+ldz*(j-1)] //z( jr, j ) = -s*z( jr, j+1 ) + c*z( jr, j )
				z[jr-1+ldz*(j)] = temp                                       //z( jr, j+1 ) = temp
			} //180          CONTINUE
		}
	} //190    CONTINUE
	jiter = jiter + 1
	goto G350
G200:
	// }//200    CONTINUE
	if ifirst+1 == ilast {

		//CALL dlasv2( t( ilast-1, ilast-1 ), t( ilast-1, ilast ),t( ilast, ilast ), b22, b11, sr, cr, sl, cl )
		b22, b11, sr, cr, sl, cl := impl.Dlasv2(t[ilast-2+ldt*(ilast-2)], t[ilast-2+ldt*(ilast-1)], t[ilast-1+ldt*(ilast-1)])
		if b11 < 0 {
			cr = -cr
			sr = -sr
			b11 = -b11
			b22 = -b22
		}

		bi.Drot(ilastm+1-ifirst, h[ilast-2+ldh*(ilast-2):], ldh, h[ilast-1+ldh*(ilast-2):], ldh, cl, sl) //bi.Drot( ilastm+1-ifirst, h( ilast-1, ilast-1 ), ldh,h( ilast, ilast-1 ), ldh, cl, sl )
		bi.Drot(ilast+1-ifrstm, h[ifrstm-1+ldh*(ilast-2):], 1, h[ifrstm-1+ldh*(ilast-1):], 1, cr, sr)    //CALL drot( ilast+1-ifrstm, h( ifrstm, ilast-1 ), 1,h( ifrstm, ilast ), 1, cr, sr )

		if ilast < ilastm {
			bi.Drot(ilastm-ilast, t[ilast-2+ldt*(ilast):], ldt, t[ilast-1+ldt*(ilast):], ldt, cl, sl) //CALL drot( ilastm-ilast, t( ilast-1, ilast+1 ), ldt,t( ilast, ilast+1 ), ldt, cl, sl )
		}
		if ifrstm < ilast-1 {
			bi.Drot(ifirst-ifrstm, t[ifrstm-1+ldt*(ilast-2):], 1, t[ifrstm-1+ldt*(ilast-1):], 1, cr, sr) //CALL drot( ifirst-ifrstm, t( ifrstm, ilast-1 ), 1,t( ifrstm, ilast ), 1, cr, sr )
		}

		if ilq {
			bi.Drot(n, q[ldq*(ilast-2):], 1, q[ldq*(ilast-1):], 1, cl, sl) //CALL drot( n, q( 1, ilast-1 ), 1, q( 1, ilast ), 1, cl,sl )

		}
		if ilz {
			bi.Drot(n, z[ldz*(ilast-2):], 1, z[ldz*(ilast-1):], 1, cr, sr) //CALL drot( n, z( 1, ilast-1 ), 1, z( 1, ilast ), 1, cr,sr )
		}
		t[ilast-2+ldt*(ilast-2)] = b11 //t( ilast-1, ilast-1 ) = b11
		t[ilast-2+ldt*(ilast-1)] = 0   //t( ilast-1, ilast ) = zero
		t[ilast-1+ldt*(ilast-2)] = 0   //t( ilast, ilast-1 ) = zero
		t[ilast-1+ldt*(ilast-1)] = b22 //t( ilast, ilast ) = b22

		if b22 < 0 {
			for j = ifrstm; j <= ilast; j++ { //DO 210 j = ifrstm, ilast
				h[j-1+ldh*(ilast-1)] = -h[j-1+ldh*(ilast-1)] //h( j, ilast ) = -h( j, ilast )
				t[j-1+ldt*(ilast-1)] = -t[j-1+ldt*(ilast-1)] //t( j, ilast ) = -t( j, ilast )
			} //210          CONTINUE

			if ilz {
				for j = 1; j <= n; j++ { //DO 220 j = 1, n
					z[j-1+ldz*(ilast-1)] = -z[j-1+ldz*(ilast-1)] //z( j, ilast ) = -z( j, ilast )
				} //220             CONTINUE
			}
			b22 = -b22
		}

		//CALL dlag2( h( ilast-1, ilast-1 ), ldh,t( ilast-1, ilast-1 ), ldt, safmath.Min*safety, s1,temp, wr, temp2, wi )
		s1, temp, wr, temp2, wi = impl.Dlag2(h[ilast-2+ldh*(ilast-2):], ldh, t[ilast-2+ldt*(ilast-2):], ldt)
		if wi == 0 {
			goto G350
		}
		s1inv = 1 / s1

		a11 = h[ilast-2+ldh*(ilast-2)] //a11 = h( ilast-1, ilast-1 )
		a21 = h[ilast-1+ldh*(ilast-2)] //a21 = h( ilast, ilast-1 )
		a12 = h[ilast-2+ldh*(ilast-1)] //a12 = h( ilast-1, ilast )
		a22 = h[ilast-1+ldh*(ilast-1)] //a22 = h( ilast, ilast )

		c11r = s1*a11 - wr*b11
		c11i = -wi * b11
		c12 = s1 * a12
		c21 = s1 * a21
		c22r = s1*a22 - wr*b22
		c22i = -wi * b22

		if math.Abs(c11r)+math.Abs(c11i)+math.Abs(c12) > math.Abs(c21)+math.Abs(c22r)+math.Abs(c22i) { //IF( math.Abs( c11r )+math.Abs( c11i )+math.Abs( c12 ) > math.Abs( c21 )+math.Abs( c22r )+math.Abs( c22i ) ) THEN
			t1 = impl.Dlapy3(c12, c11r, c11i)
			cz = c12 / t1
			szr = -c11r / t1
			szi = -c11i / t1
		} else {
			cz = impl.Dlapy2(c22r, c22i)
			if cz <= safmin {
				cz = 0
				szr = 1
				szi = 0
			} else {
				tempr = c22r / cz
				tempi = c22i / cz
				t1 = impl.Dlapy2(cz, c21)
				cz = cz / t1
				szr = -c21 * tempr / t1
				szi = c21 * tempi / t1
			}
		}

		an = math.Abs(a11) + math.Abs(a12) + math.Abs(a21) + math.Abs(a22)
		bn = math.Abs(b11) + math.Abs(b22)
		wabs = math.Abs(wr) + math.Abs(wi)
		if s1*an > wabs*bn {
			cq = cz * b11
			sqr = szr * b22
			sqi = -szi * b22
		} else {
			a1r = cz*a11 + szr*a12
			a1i = szi * a12
			a2r = cz*a21 + szr*a22
			a2i = szi * a22
			cq = impl.Dlapy2(a1r, a1i)
			if cq <= safmin {
				cq = 0
				sqr = 1
				sqi = 0
			} else {
				tempr = a1r / cq
				tempi = a1i / cq
				sqr = tempr*a2r + tempi*a2i
				sqi = tempi*a2r - tempr*a2i
			}
		}
		t1 = impl.Dlapy3(cq, sqr, sqi)
		cq = cq / t1
		sqr = sqr / t1
		sqi = sqi / t1

		tempr = sqr*szr - sqi*szi
		tempi = sqr*szi + sqi*szr
		b1r = cq*cz*b11 + tempr*b22
		b1i = tempi * b22
		b1a = impl.Dlapy2(b1r, b1i)
		b2r = cq*cz*b22 + tempr*b11
		b2i = -tempi * b11
		b2a = impl.Dlapy2(b2r, b2i)

		beta[ilast-2] = b1a                   //beta( ilast-1 ) = b1a
		beta[ilast-1] = b2a                   //beta( ilast ) = b2a
		alphar[ilast-2] = (wr * b1a) * s1inv  //alphar( ilast-1 ) = ( wr*b1a )*s1inv
		alphai[ilast-2] = (wi * b1a) * s1inv  //alphai( ilast-1 ) = ( wi*b1a )*s1inv
		alphar[ilast-1] = (wr * b2a) * s1inv  //alphar( ilast ) = ( wr*b2a )*s1inv
		alphai[ilast-1] = -(wi * b2a) * s1inv //alphai( ilast ) = -( wi*b2a )*s1inv

		ilast = ifirst - 1
		if ilast < ilo {
			goto G380
		}

		iiter = 0
		eshift = 0
		if !ilschr {
			ilastm = ilast
			if ifrstm > ilast {
				ifrstm = ilo
			}
		}
		goto G350
	} else {

		//ad11 = ( ascale*h( ilast-1, ilast-1 ) ) /( bscale*t( ilast-1, ilast-1 ) )
		//ad21 = ( ascale*h( ilast, ilast-1 ) ) /( bscale*t( ilast-1, ilast-1 ) )
		//ad12 = ( ascale*h( ilast-1, ilast ) ) /( bscale*t( ilast, ilast ) )
		//ad22 = ( ascale*h( ilast, ilast ) ) /( bscale*t( ilast, ilast ) )
		//u12 = t( ilast-1, ilast ) / t( ilast, ilast )
		//ad11l = ( ascale*h( ifirst, ifirst ) ) /( bscale*t( ifirst, ifirst ) )
		//ad21l = ( ascale*h( ifirst+1, ifirst ) ) /( bscale*t( ifirst, ifirst ) )
		//ad12l = ( ascale*h( ifirst, ifirst+1 ) ) /( bscale*t( ifirst+1, ifirst+1 ) )
		//ad22l = ( ascale*h( ifirst+1, ifirst+1 ) ) /( bscale*t( ifirst+1, ifirst+1 ) )
		//ad32l = ( ascale*h( ifirst+2, ifirst+1 ) ) /( bscale*t( ifirst+1, ifirst+1 ) )
		//u12l = t( ifirst, ifirst+1 ) / t( ifirst+1, ifirst+1 )
		ad11 = (ascale * h[ilast-2+ldh*(ilast-2)]) / (bscale * t[ilast-2+ldt*(ilast-2)])
		ad21 = (ascale * h[ilast-1+ldh*(ilast-2)]) / (bscale * t[ilast-2+ldt*(ilast-2)])
		ad12 = (ascale * h[ilast-2+ldh*(ilast-1)]) / (bscale * t[ilast-1+ldt*(ilast-1)])
		ad22 = (ascale * h[ilast-1+ldh*(ilast-1)]) / (bscale * t[ilast-1+ldt*(ilast-1)])
		u12 = t[ilast-2+ldt*(ilast-1)] / t[ilast-1+ldt*(ilast-1)]
		ad11l = (ascale * h[ifirst-1+ldh*(ifirst-1)]) / (bscale * t[ifirst-1+ldt*(ifirst-1)])
		ad21l = (ascale * h[ifirst+ldh*(ifirst-1)]) / (bscale * t[ifirst-1+ldt*(ifirst-1)])
		ad12l = (ascale * h[ifirst-1+ldh*(ifirst)]) / (bscale * t[ifirst+ldt*(ifirst)])
		ad22l = (ascale * h[ifirst+ldh*(ifirst)]) / (bscale * t[ifirst+ldt*(ifirst)])
		ad32l = (ascale * h[ifirst+1+ldh*(ifirst)]) / (bscale * t[ifirst+ldt*(ifirst)])
		u12l = t[ifirst-1+ldt*(ifirst)] / t[ifirst+ldt*(ifirst)]
		v[0] = (ad11-ad11l)*(ad22-ad11l) - ad12*ad21 + ad21*u12*ad11l + (ad12l-ad11l*u12l)*ad21l   //v( 1 ) = ( ad11-ad11l )*( ad22-ad11l ) - ad12*ad21 +ad21*u12*ad11l + ( ad12l-ad11l*u12l )*ad21l
		v[1] = ((ad22l - ad11l) - ad21l*u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21*u12) * ad21l //v( 2 ) = ( ( ad22l-ad11l )-ad21l*u12l-( ad11-ad11l )-( ad22-ad11l )+ad21*u12 )*ad21l
		v[2] = ad32l * ad21l                                                                       //v( 3 ) = ad32l*ad21l

		istart = ifirst
		var dump float64
		dump, tau = impl.Dlarfg(3, v[0], v[1:], 1) //CALL dlarfg( 3, v( 1 ), v( 2 ), 1, tau )
		_ = dump
		v[0] = 1 //v( 1 ) = one

		for j = istart; j <= ilast-2; j++ { //DO 290 j = istart, ilast - 2

			if j > istart {
				//v( 1 ) = h( j, j-1 )
				//v( 2 ) = h( j+1, j-1 )
				//v( 3 ) = h( j+2, j-1 )
				v[0] = h[j-1+ldh*(j-2)]
				v[1] = h[j+ldh*(j-2)]
				v[2] = h[j+1+ldh*(j-2)]

				//beta, tau = impl.Dlarfg(3, h[j-1+ldh*(j-2)], v[1], 1) //CALL dlarfg( 3, h( j, j-1 ), v( 2 ), 1, tau )
				v[0] = 1             //v( 1 ) = one
				h[j+ldh*(j-2)] = 0   //h( j+1, j-1 ) = zero
				h[j+1+ldh*(j-2)] = 0 //h( j+2, j-1 ) = zero
			}

			for jc = j; jc <= ilast; jc++ { //DO 230 jc = j, ilastm
				temp = tau * (h[j-1+ldh*(jc-1)] + v[1]*h[j+ldh*(jc-1)] + v[2]*h[j+1+ldh*(jc-1)])  //temp = tau*( h( j, jc )+v( 2 )*h( j+1, jc )+v( 3 )*h( j+2, jc ) )
				h[j-1+ldh*(jc-1)] = h[j-1+ldh*(jc-1)] - temp                                      //h( j, jc ) = h( j, jc ) - temp
				h[j+ldh*(jc-1)] = h[j+ldh*(jc-1)] - temp*v[1]                                     //h( j+1, jc ) = h( j+1, jc ) - temp*v( 2 )
				h[j+1+ldh*(jc-1)] = h[j+1+ldh*(jc-1)] - temp*v[2]                                 //h( j+2, jc ) = h( j+2, jc ) - temp*v( 3 )
				temp2 = tau * (t[j-1+ldt*(jc-1)] + v[1]*t[j+ldh*(jc-1)] + v[2]*t[j+1+ldh*(jc-1)]) //temp2 = tau*( t( j, jc )+v( 2 )*t( j+1, jc )+v( 3 )*t( j+2, jc ) )
				t[j-1+ldt*(jc-1)] = t[j-1+ldt*(jc-1)] - temp2                                     //t( j, jc ) = t( j, jc ) - temp2
				t[j+ldt*(jc-1)] = t[j+ldt*(jc-1)] - temp2*v[1]                                    //t[j+1+ldt*(jc-1)] = t( j+1, jc ) - temp2*v( 2 )
				t[j+1+ldt*(jc-1)] = t[j+1+ldt*(jc-1)] - temp2*v[2]                                //t( j+2, jc ) = t( j+2, jc ) - temp2*v( 3 )
			} //230          CONTINUE
			if ilq {
				for jr = 1; jr <= n; jr++ { //DO 240 jr = 1, n
					temp = tau * (q[jr-1+ldq*(j-1)] + v[1]*q[jr-1+ldq*(j)] + v[2]*q[jr-1+ldq*(j+1)]) //temp = tau*( q( jr, j )+v( 2 )*q( jr, j+1 )+v( 3 )*q( jr, j+2 ) )
					q[jr-1+ldq*(j-1)] = q[jr-1+ldq*(j-1)] - temp                                     //q( jr, j ) = q( jr, j ) - temp
					q[jr-1+ldq*(j)] = q[jr-1+ldq*(j)] - temp*v[1]                                    //q( jr, j+1 ) = q( jr, j+1 ) - temp*v( 2 )
					q[jr-1+ldq*(j+1)] = q[jr-1+ldq*(j+1)] - temp*v[2]                                //q( jr, j+2 ) = q( jr, j+2 ) - temp*v( 3 )
				} //240             CONTINUE
			}

			ilpivt = false
			temp = math.Max(math.Abs(t[j+ldt*(j)]), math.Abs(t[j+ldt*(j+1)]))      //temp = math.Max( math.Abs( t( j+1, j+1 ) ), math.Abs( t( j+1, j+2 ) ) )
			temp2 = math.Max(math.Abs(t[j+1+ldt*(j)]), math.Abs(t[j+1+ldt*(j+1)])) //temp2 = math.Max( math.Abs( t( j+2, j+1 ) ), math.Abs( t( j+2, j+2 ) ) )
			if math.Max(temp, temp2) < safmin {                                    //IF( math.Max( temp, temp2 ) < safmath.Min ) THEN
				scale = 0
				u1 = 1
				u2 = 0
				goto G250
			} else if temp >= temp2 {
				w11 = t[j+ldt*(j)]     //w11 = t( j+1, j+1 )
				w21 = t[j+1+ldt*(j)]   //w21 = t( j+2, j+1 )
				w12 = t[j+ldt*(j+1)]   //w12 = t( j+1, j+2 )
				w22 = t[j+1+ldt*(j+1)] //w22 = t( j+2, j+2 )
				u1 = t[j+ldt*(j-1)]    //u1 = t( j+1, j )
				u2 = t[j+1+ldt*(j-1)]  //u2 = t( j+2, j )
			} else {
				w21 = t[j+ldt*(j)]     //w21 = t( j+1, j+1 )
				w11 = t[j+1+ldt*(j)]   //w11 = t( j+2, j+1 )
				w22 = t[j+ldt*(j+1)]   //w22 = t( j+1, j+2 )
				w12 = t[j+1+ldt*(j+1)] //w12 = t( j+2, j+2 )
				u2 = t[j+ldt*(j-1)]    //u2 = t( j+1, j )
				u1 = t[j+1+ldt*(j-1)]  //u1 = t( j+2, j )
			}

			if math.Abs(w12) > math.Abs(w11) {
				ilpivt = true
				temp = w12
				temp2 = w22
				w12 = w11
				w22 = w21
				w11 = temp
				w21 = temp2
			}

			temp = w21 / w11
			u2 = u2 - temp*u1
			w22 = w22 - temp*w12
			w21 = 0

			scale = 1
			if math.Abs(w22) < safmin {
				scale = 0
				u2 = 1
				u1 = -w12 / w11
				goto G250
			}
			if math.Abs(w22) < math.Abs(u2) {
				scale = math.Abs(w22 / u2)
			}
			if math.Abs(w11) < math.Abs(u1) {
				scale = math.Min(scale, math.Abs(w11/u1))
			}

			u2 = (scale * u2) / w22
			u1 = (scale*u1 - w12*u2) / w11
		G250:
			//} //250          CONTINUE
			if ilpivt {
				temp = u2
				u2 = u1
				u1 = temp
			}

			t1 = math.Sqrt(scale*scale + u1*u1 + u2*u2) //no llega t1 = sqrt( scale**2+u1**2+u2**2 )
			tau = 1 + scale/t1
			vs = -1 / (scale + t1)
			v[0] = 1       //v( 1 ) = 1
			v[1] = vs * u1 //v( 2 ) = vs*u1
			v[2] = vs * u2 //v( 3 ) = vs*u2

			for jr = ifrstm; jr <= min(j+3, ilast); jr++ { //DO 260 jr = ifrstm, math.Min( j+3, ilast )
				temp = tau * (h[jr-1+ldh*(j-1)] + v[1]*h[jr-1+ldh*(j)] + v[2]*h[jr-1+ldh*(j+1)]) //temp = tau*( h( jr, j )+v( 2 )*h( jr, j+1 )+v( 3 )*h( jr, j+2 ) )
				h[jr-1+ldh*(j-1)] = h[jr-1+ldh*(j-1)] - temp                                     //h( jr, j ) = h( jr, j ) - temp
				h[jr-1+ldh*(j)] = h[jr-1+ldh*(j)] - temp*v[1]                                    //h( jr, j+1 ) = h( jr, j+1 ) - temp*v( 2 )
				h[jr-1+ldh*(j+1)] = h[jr-1+ldh*(j+1)] - temp*v[2]                                //h( jr, j+2 ) = h( jr, j+2 ) - temp*v( 3 )
			} //260          CONTINUE
			for jr = ifrstm; jr <= j+2; jr++ { //  DO 270 jr = ifrstm, j + 2
				temp = tau * (t[jr-1+ldt*(j-1)] + v[1]*t[jr-1+ldt*(j)] + v[2]*t[jr-1+ldt*(jr+1)]) //temp = tau*( t( jr, j )+v( 2 )*t( jr, j+1 )+v( 3 )*t( jr, j+2 ) )
				t[jr-1+ldt*(j-1)] = t[jr-1+ldt*(j-1)] - temp                                      //t( jr, j ) = t( jr, j ) - temp
				t[jr-1+ldt*(j)] = t[jr-1+ldt*(j)] - temp*v[1]                                     //t( jr, j+1 ) = t( jr, j+1 ) - temp*v( 2 )
				t[jr-1+ldt*(j+1)] = t[jr-1+ldt*(j+1)] - temp*v[2]                                 //t( jr, j+2 ) = t( jr, j+2 ) - temp*v( 3 )
			} //270          CONTINUE
			if ilz {
				jr = 1
			G900:
				if jr <= n { //for jr = 1; jr <= n; n++ { //DO 280 jr = 1, n
					temp = tau * (z[jr-1+ldz*(j-1)] + v[1]*z[jr-1+ldz*(j)] + v[2]*z[jr-1+ldz*(j+1)]) //temp = tau*( z( jr, j )+v( 2 )*z( jr, j+1 )+v( 3 )*z( jr, j+2 ) )
					z[jr-1+ldz*(j-1)] = z[jr-1+ldz*(j-1)] - temp                                     //z( jr, j ) = z( jr, j ) - temp
					z[jr-1+ldz*(j)] = z[jr-1+ldz*(j)] - temp*v[1]                                    //z( jr, j+1 ) = z( jr, j+1 ) - temp*v( 2 )
					z[jr-1+ldz*(j+1)] = z[jr-1+ldz*(j+1)] - temp*v[2]                                //z( jr, j+2 ) = z( jr, j+2 ) - temp*v( 3 )
					jr = jr + 1
					goto G900
				} //280             CONTINUE
			}
			t[j+ldz*(j-1)] = 0   //t( j+1, j ) = zero
			t[j+1+ldz*(j-1)] = 0 //t( j+2, j ) = zero
		} //290       CONTINUE

		j = ilast - 1
		temp = h[j-1+ldh*(j-2)]                                     //temp = h( j, j-1 )
		c, s, h[j-1+ldh*(j-2)] = impl.Dlartg2(temp, h[j+ldh*(j-2)]) //CALL dlartg( temp, h( j+1, j-1 ), c, s, h( j, j-1 ) )
		h[j+ldh*(j-2)] = 0                                          //h( j+1, j-1 ) = zero

		for jc = j; jc <= ilastm; jc++ { //DO 300 jc = j, ilastm
			temp = c*h[j-1+ldh*(jc-1)] + s*h[j+ldh*(jc-1)]             //temp = c*h( j, jc ) + s*h( j+1, jc )
			h[j+ldh*(jc-1)] = -s*h[j-1+ldh*(jc-1)] + c*h[j+ldh*(jc-1)] //h( j+1, jc ) = -s*h( j, jc ) + c*h( j+1, jc )
			h[j-1+ldh*(jc-1)] = temp                                   //h( j, jc ) = temp
			temp2 = c*t[j-1+ldt*(jc-1)] + s*t[j+ldt*(jc-1)]            //temp2 = c*t( j, jc ) + s*t( j+1, jc )
			t[j+ldt*(jc-1)] = -s*t[j-1+ldt*(jc-1)] + c*t[j+ldt*(jc-1)] //t( j+1, jc ) = -s*t( j, jc ) + c*t( j+1, jc )
			t[j-1+ldt*(jc-1)] = temp2                                  //t( j, jc ) = temp2
		} //300       CONTINUE
		if ilq {
			for j = 1; j <= n; j++ { //DO 310 jr = 1, n
				temp = c*q[jr-1+ldq*(j-1)] + s*q[jr-1+ldq*(j)]             //temp = c*q( jr, j ) + s*q( jr, j+1 )
				q[jr-1+ldq*(j)] = -s*q[jr-1+ldq*(j-1)] + c*q[jr-1+ldq*(j)] //q( jr, j+1 ) = -s*q( jr, j ) + c*q( jr, j+1 )
				q[jr-1+ldq*(j-1)] = temp                                   //q( jr, j ) = temp
			} //310          CONTINUE
		}

		temp = t[j+ldt*(j)]                                     //temp = t( j+1, j+1 )
		c, s, t[j+ldt*(j)] = impl.Dlartg2(temp, t[j+ldt*(j-1)]) //CALL dlartg( temp, t( j+1, j ), c, s, t( j+1, j+1 ) )
		t[j+ldt*(j-1)] = 0                                      //t( j+1, j ) = zero

		for jr = ifrstm; jr <= ilast; jr++ { //DO 320 jr = ifrstm, ilast
			temp = c*h[jr-1+ldh*(j)] + s*h[jr-1+ldh*(j-1)]               //temp = c*h( jr, j+1 ) + s*h( jr, j )
			h[jr-1+ldh*(j-1)] = -s*h[jr-1+ldh*(j)] + c*h[jr-1+ldh*(j-1)] //h( jr, j ) = -s*h( jr, j+1 ) + c*h( jr, j )
			h[jr-1+ldh*(j)] = temp                                       //h( jr, j+1 ) = temp
		} //320       CONTINUE
		for jr = ifrstm; jr <= ilast-1; jr++ { //DO 330 jr = ifrstm, ilast - 1
			temp = c*t[jr-1+ldt*(j)] + s*t[jr-1+ldt*(j-1)]               //temp = c*t( jr, j+1 ) + s*t( jr, j )
			t[jr-1+ldt*(j-1)] = -s*t[jr-1+ldt*(j)] + c*t[jr-1+ldt*(j-1)] //t( jr, j ) = -s*t( jr, j+1 ) + c*t( jr, j )
			t[jr-1+ldt*(j)] = temp                                       //t( jr, j+1 ) = temp
		} //330       CONTINUE
		if ilz {
			for jr = 1; jr <= n; jr++ { //DO 340 jr = 1, n
				temp = c*z[jr-1+ldz*(j)] + s*z[jr-1+ldz*(j-1)]               //temp = c*z( jr, j+1 ) + s*z( jr, j )
				z[jr-1+ldz*(j-1)] = -s*z[jr-1+ldz*(j)] + c*z[jr-1+ldz*(j-1)] //z( jr, j ) = -s*z( jr, j+1 ) + c*z( jr, j )
				z[jr-1+ldz*(j)] = temp                                       //z( jr, j+1 ) = temp
			} //340          CONTINUE
		}
	}
	jiter = jiter + 1
	goto G350
	//G350:
	//G360:
	//}//350    CONTINUE
	//}//360 CONTINUE

	//info = ilast
	//goto G420
G380:
	//}//380 CONTINUE

	for j = 1; j <= ilo-1; j++ { //DO 410 j = 1, ilo - 1
		if t[j-1+ldt*(j-1)] < 0 { //IF( t( j, j ) < zero ) THEN
			if ilschr {
				for jr = 1; jr <= j; jr++ { //DO 390 jr = 1, j
					h[jr-1+ldh*(j-1)] = -h[jr-1+ldh*(j-1)] //h( jr, j ) = -h( jr, j )
					t[jr-1+ldt*(j-1)] = -t[jr-1+ldt*(j-1)] //t( jr, j ) = -t( jr, j )
				} //390          CONTINUE
			} else {
				h[j-1+ldh*(j-1)] = -h[j-1+ldh*(j-1)] //h( j, j ) = -h( j, j )
				t[j-1+ldt*(j-1)] = -t[j-1+ldt*(j-1)] //t( j, j ) = -t( j, j )
			}
			if ilz {
				for jr = 1; jr <= n; jr++ { //DO 400 jr = 1, n
					z[jr-1+ldz*(j-1)] = -z[jr-1+ldz*(j-1)] //z( jr, j ) = -z( jr, j )
				} //400          CONTINUE
			}
		}
		alphar[j-1] = h[j-1+ldh*(j-1)] //alphar( j ) = h( j, j )
		alphai[j-1] = 0                //alphai( j ) = zero
		beta[j-1] = t[j-1+ldt*(j-1)]   //beta( j ) = t( j, j )
	} //410 CONTINUE

	info = 0
G420:
	//}// 420 CONTINUE
	work[0] = float64(n - 1) //work( 1 ) = dble( n )
	return

}
