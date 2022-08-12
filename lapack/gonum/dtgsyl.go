package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dpstrf computes the Cholesky factorization with complete pivoting of an n×n
// symmetric positive semidefinite matrix A.
//
// The factorization has the form
//  Pᵀ * A * P = Uᵀ * U ,  if uplo = blas.Upper,
//  Pᵀ * A * P = L  * Lᵀ,  if uplo = blas.Lower,
// where U is an upper triangular matrix and L is lower triangular, and P is
// stored as vector piv.
//
// Dpstrf does not attempt to check that A is positive semidefinite.
//
// The length of piv must be n and the length of work must be at least 2*n,
// otherwise Dpstrf will panic.
//
// Dpstrf is an internal routine. It is exported for testing purposes.

// IWORK is INTEGER array, dimension (M+N+6)
//func (impl Implementation) Dtgsyl(trans blas.Transpose, ijob int, m int, n int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, d []float64, ldd int, e []float64, lde int, f []float64, ldf int, dif float64, scale float64, work []float64, lwork int, iwork []float64, info int) int {

func (impl Implementation) Dtgsyl(trans blas.Transpose, ijob int, m int, n int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, d []float64, ldd int, e []float64, lde int, f []float64, ldf int, work []float64, lwork int, iwork []int, info int) (dif, scale float64) {

	//parameter( zero == 0, one == 1 )
	var lquery, notran bool
	//var i, ie, ifunc, iround, is, isolve, j, je, js, k int
	var i, ie, is, isolve, j, je, js int
	var linfo, lwmin, mb, nb, p, ppqq, pq, q int
	var scale2, scaloc float64
	var dsum, dscale float64
	ipiv := make([]int, 8, 8)
	jpiv := make([]int, 8, 8)
	//variables de salida
	//var dif float64
	//var scale float64
	//.
	//for x = 0; x <= (m + n + 6); x++ {
	//	iwork = append(iwork, nil)
	//}
	info = 0
	notran = (trans == blas.NoTrans) //como es el tema de los tipos? blas.uplo cuando es n? y blas.all cuando es t??
	lquery = (lwork == -1)
	bi := blas64.Implementation()
	if !(notran) && !(trans == blas.Trans) {
		panic(badTrans)
	} else if notran {
		if (ijob < 0) || (ijob > 4) {
			panic(-2)
		}
	}

	if info == 0 {
		if m <= 0 {
			panic(mLT0) // usar panic
		} else if n <= 0 {
			panic(nLT0)
		} else if lda < max(1, m) {
			panic(badLdA)
		} else if ldb < max(1, n) {
			panic(badLdB)
		} else if ldc < max(1, m) {
			panic(badLdC)
		} else if ldd < max(1, m) {
			//panic(badLdD)
		} else if lde < max(1, n) {
			//panic(badLdE)
		} else if ldf < max(1, m) {
			panic(badLdF)
		}
	}
	if info == 0 {
		if notran {
			if (ijob == 1) || (ijob == 2) {
				lwmin = max(1, 2*m*n)
			} else {
				lwmin = 1
			}
		} else {
			lwmin = 1
		}

		//work[1] = lwmin
		//work[0] = float64(lwmin)
		work = make([]float64, 1) //corregir aca usar append
		work[0] = float64(lwmin)
		if (lwork < lwmin) && !(lquery) {
			info = -20
		}
	}

	if info != 0 {
		//xerbla( "DTGSYL", -info )
		//RETURN ;
		panic(-info)
	} else if lquery {
		panic(lquery)
	}
	//     Quick return if possible.

	if (m == 0) || (n == 0) {
		//scale[0] = float64(1)
		scale = (1)
		if notran {
			if ijob <= 0 {
				//dif[0] = float64(0)
				dif = (0)
			}
		}
		//RETURN
	}

	//    Determine optimal block sizes MB and NB.
	var opts string
	if trans == blas.Trans {
		opts += "T"
	} else {
		opts += "N"
	}
	//mb = impl.Ilaenv(2, "DTGSYL", opts, m, n, -1, -1) //aca que onda con trans porque la funcion dice que es lugar para opciones del algoritmo.
	//nb = impl.Ilaenv(5, "DTGSYL", opts, m, n, -1, -1) //aca que onda con trans porque la funcion dice que es lugar para opciones del algoritmo.
	mb = 2
	nb = 2
	isolve = 1

	ifunc := 0
	_ = ifunc
	if notran {
		if ijob >= 3 {
			ifunc = ijob - 2
			//impl.Dlaset('F', m, n, 0, 0, c, ldc)
			impl.Dlaset(blas.All, m, n, 0, 0, f, ldf) //falta el tipo F
		} else if ijob >= 1 {
			isolve = 2
		}
	}

	if ((mb <= 1) && (nb <= 1)) || ((mb >= m) && (nb >= n)) {
		//DO 30 iround = 1, isolve
		for iround := 1; iround <= isolve; iround++ {
			//           Use unblocked Level 2 solver.
			dscale = 0
			dsum = 1
			pq = 0
			//impl.Dtgsy2(trans, ifunc, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, scale, dsum, dscale, iwork, pq, info)
			if dscale != 0 {
				if (ijob == 1) || (ijob == 3) {
					//dif = Sqrt(dble(2*m*n)) / (dscale * Sqrt(dsum))
					dif = math.Sqrt(float64(2*m*n) / (dscale * math.Sqrt(dsum)))
				} else {
					//dif = math.Sqrt(dble(pq)) / (dscale * math.Sqrt(dsum))
					dif = math.Sqrt((float64(pq))) / (dscale * math.Sqrt(dsum))
				}
			}

			if (isolve == 2) && (iround == 1) {
				if notran {
					ifunc = ijob
				}
				scale2 = scale
				impl.Dlacpy(blas.All, m, n, c, ldc, work, m)
				impl.Dlacpy(blas.All, m, n, f, ldf, work[m*n:], m) // -1 porque se usan variables longitud en el indice
				impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
				impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
			} else if (isolve == 2) && (iround == 2) {
				impl.Dlacpy(blas.All, m, n, work, m, c, ldc)
				impl.Dlacpy(blas.All, m, n, work[m*n:], m, f, ldf)
				scale = scale2
			}
			//cierro for.
		} //cierro for.
		//*retrun ????.
	}

	//     Determine block structure of A.

	p = 0 //tener en cuenta se le resto 1
	i = 1 //tener en cuenta se le resto 1
G40: //CONTINUE
	if i > m {
		goto G50
	}

	p = p + 1
	if ((len(iwork) - 1) < p) || (len(iwork) == 0) {
		//iwork = append(iwork, (p - len(iwork)))
		iwork = append(iwork, 0)
	}

	iwork[p-1] = i //dimensionar iwork            aca quede debugear

	i += mb
	if i >= m { //tener en cuenta se le resto 1
		goto G50
	}
	if a[i-1+lda*(i-2)] != 0 { ///a[(i*(lda+1))-(lda+2)]//a(i,i-1)  5n-6 6n-7
		i = i + 1
	}
	goto G40
G50: //CONTINUE

	if ((len(iwork) - 1) < (p + 1)) || (len(iwork) == 0) {
		iwork = append(iwork, 0)
	}
	iwork[p] = (m + 1) // tener en cuenta fijarse si iwork es indice y si es correcto hacerle el +1.
	if iwork[p-1] == iwork[p] {
		p--
	}

	//     Determine block structure of B.
	if iwork[p-1] == iwork[p] {
		p = p - 1
	}
	q = p + 1 //tener en cuenta si son indices, puede que haya que corregir.
	j = 1     //tener en cuenta si son indices, puede que haya que corregir.
G60: //CONTINUE
	if j > n {
		goto G70
	}
	q = q + 1
	if ((len(iwork) - 1) < q) || (len(iwork) == 0) {
		iwork = append(iwork, 0)
	}
	iwork[q-1] = j //tengo q hace una funcion append
	j = j + nb
	if j >= n {
		goto G70
	}
	//original b[j][j-1]   corregida b[(j*(ldb+1))-(ldb+2)]   propuesta b[j*ldb+j-1]
	if b[j-1+ldb*(j-2)] != 0 {
		j = j + 1
	}
	goto G60
G70: //CONTINUE

	if ((len(iwork) - 1) < (q + 1)) || (len(iwork) == 0) {
		iwork = append(iwork, 0)
	}

	iwork[q] = n + 1
	if iwork[q-1] == iwork[q] {
		q--
	}
	if notran {
		var iround int
		//DO 150 iround = 1, isolve
		for iround = 1; iround == isolve; iround++ {
			//           Solve (I, J)-subsystem
			//              A(I, I) * R(I, J) - L(I, J) * B(J, J) = C(I, J)
			//              D(I, I) * R(I, J) - L(I, J) * E(J, J) = F(I, J)
			//          for I = P, P - 1,..., 1; J = 1, 2,..., Q .
			dscale = 0
			dsum = 1
			pq = 0
			scale = 1
			//DO 130 j = p + 2, q
			for j := (p + 2); j <= q; j++ {
				js = int(iwork[j-1])
				je = int(iwork[j] - 1)
				nb = je - js + 1
				//DO 120 i = p, 1, -1    entra con i=6 p=2 y despues salta a i=4
				for i := p; i >= 1; i-- { //aca no entro i=2 p=2 en fortran
					is = (iwork[i-1])
					ie = (iwork[i] - 1)
					mb = ie - is + 1
					ppqq = 0

					var ijob2 lapack.MaximizeNormXJob
					ijob2 = 0
					//DTGSY2( TRANS         , IJOB , M ,  N, A                      , LDA, B                   , LDB, C                      , LDC,D                      ,LDD , E                      , LDE, F                     , LDF, SCALE ,RDSUM, RDSCAL,IWORK       ,   PQ, INFO )
					//tgsy2(           trans, ifunc, mb, nb, a( is, is )            , lda,b( js, js )          , ldb, c( is, js )            , ldc,d( is, is )            , ldd, e( js, js )            , lde,f( is, js )            , ldf, scaloc, dsum, dscale,iwork( q+2 ), ppqq, linfo )
					var scalout, sumout float64
					_ = scalout
					_ = sumout //blas.NoTrans es lo q originalmente va
					//scaloc, scalout, sumout, ppqq, linfo = impl.Dtgsy2(trans, ijob2, mb, nb, a[(lda*is-(lda+1)+is):], lda, b[(ldb*js-(ldb+1)+js):], ldb, c[(ldc*is-(ldc+1)+js):], ldc, d[(ldd*is-(ldd+1)+is):], ldd, e[(lde*js-(lde+1)+js):], lde, f[(ldf*is-(ldf+1)+js):], ldf, dsum, dscale, iwork[(q+2):]) // la matriz c no de queda igual que en el fortran y linfo tampoco
					scaloc, scalout, sumout, ppqq, linfo = impl.Dtgsy2(trans, ijob2, mb, nb, a[(is-1+lda*(is-1)):], lda, b[(js-1+ldb*(js-1)):], ldb, c[(is-1+ldc*(js-1)):], ldc, d[is-1+ldb*(is-1):], ldd, e[js-1+lde*(js-1):], lde, f[is-1+ldf*(js-1):], ldf, dsum, dscale, iwork[:(q+1)], ipiv, jpiv) // CALL dtgsy2( trans, ifunc, mb, nb, a( is, is ), lda,b( js, js ), ldb, c( is, js ), ldc,d( is, is ), ldd, e( js, js ), lde,f( is, js ), ldf, scaloc, dsum, dscale,iwork( q+2 ), ppqq, linfo )
					//   Dtgsy2(trans blas. , ijob , m , n , a []float64            ,lda ,b []float64          , ldb, c []float64            , ldc,d []float64            , ldd, e []float64            , lde, f []float64           , ldf, rdsum ,rdscal, iwork []int)     (scale, scalout, sumout float64, pq, info int) {

					//       impl.Dtgsy2(blas.NoTrans, ijob2, mb, nb, a[(lda+1)*is-(lda+1)]  , lda,b[(ldb+1)*js-(ldb+1)], ldb, c[(ldc*is-(ldc+1) +js)], ldc,d[(ldd*is-(ldd+1) +is)], ldd, e[(lde*js-(lde+1) +js)], lde,f[(ldf*is-(ldf+1) +js)], ldf, dsum  , dscale,iwork[q+2] )
					if linfo > 0 {
						info = linfo
					}
					pq = pq + ppqq
					if scaloc != float64(1) {
						//DO 80 k = 1, js - 1
						//for k := 1; k <= js; js-- { ////revisar aca quede
						for k := 0; k <= js; k++ {
							// CALL dscal( m, scaloc, c( 1, k ), 1 )
							// CALL dscal( m, scaloc, f( 1, k ), 1 )
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc)
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf)
							//80                CONTINUE
						}
						//DO 90 k = js, je
						for k := js; k <= je; je++ {
							//CALL dscal( is-1, scaloc, c( 1, k ), 1 )
							//CALL dscal( is-1, scaloc, f( 1, k ), 1 )
							bi.Dscal(is-1, scaloc, c[ldc*(k-1):], ldc)
							bi.Dscal(is-1, scaloc, f[ldf*(k-1):], ldf)
							//90                CONTINUE
						}
						//DO 100 k = js, je
						for k := js; k <= js; je++ {
							//bi.Dscal(m-ie, scaloc, c(ie+1, k), 1)
							//bi.Dscal(m-ie, scaloc, f(ie+1, k), 1)
							bi.Dscal(m-ie, scaloc, c[ie+ldc*(k-1):], 1)
							bi.Dscal(m-ie, scaloc, f[ie+ldf*(k-1):], 1)
							//100                CONTINUE
						}
						//DO 110 k = je + 1, n
						for k := je + 1; k < n; k++ {
							//bi.Dscal(m, scaloc, c(1, k), 1)
							//bi.Dscal(m, scaloc, f(1, k), 1)
							bi.Dscal(m, scaloc, c[ldc*(k-1):], ldc)
							bi.Dscal(m, scaloc, f[ldf*(k-1):], ldf)
							//110                CONTINUE
						}
						scale *= scaloc
					}
					//                 Substitute R(I, J) and L(I, J) into remaining
					//                 equation.
					if (i) > 1 {
						//bi.Dgemm('N', 'N', is-1, nb, mb, -1, a[1][is] , lda, c[is][js], ldc, 1, c[1][js], ldc)
						//bi.Dgemm('N', 'N', is-1, nb, mb, -1, d[1][is], ldd, c[is][js], ldc, 1, f(1, js), ldf)
						zaux := make([]float64, ldc*ldc)
						var ix, jx int
						for ix = 1; ix < ldc; ix++ {
							for jx = 1; jx < ldc; jx++ {
								//zaux[jx][ix] = z[ix][jx]
								zaux[jx-1+ldc*(ix-1)] = a[ix-1+ldc*(jx-1)]
							}
						}
						//bi.Dgemm(blas.Trans, blas.NoTrans, is-1, nb, mb, -1, a[lda*(is-1):], lda, c[is-1+ldc*(js-1)+3:], ldc, 1, c[ldc*(js-1):], ldc)
						//bi.Dgemm(blas.NoTrans, blas.NoTrans, is-1, nb, mb, -1, d[ldd*(is-1):], ldd, c[is-1+ldc*(js-1):], ldc, 1, f[ldf*(js-1):], ldf)
						impl.Dgemm2('N', 'N', is-1, nb, mb, -1, a[lda*(is-1):], lda, c[is-1+ldc*(js-1):], ldc, 1, c[ldc*(js-1):], ldc)
						impl.Dgemm2('N', 'N', is-1, nb, mb, -1, d[ldd*(is-1):], ldd, c[is-1+ldc*(js-1):], ldc, 1, f[ldf*(js-1):], ldf)

					}
					if j < q {
						//CALL dgemm( 'N', 'N', mb, n-je, nb, one,f( is, js ), ldf, b( js, je+1 ), ldb,one, c( is, je+1 ), ldc )
						//CALL dgemm( 'N', 'N', mb, n-je, nb, one,f( is, js ), ldf, e( js, je+1 ), lde,one, f( is, je+1 ), ldf )

						impl.Dgemm2('N', 'N', mb, n-je, nb, 1, f[is-1+ldf*(js-1):], ldf, b[js-1+ldb*(je):], ldb, 1, c[is-1+ldc*(je):], ldc)
						impl.Dgemm2('N', 'N', mb, n-je, nb, 1, f[is-1+ldf*(js-1):], ldf, e[js-1+lde*(je):], lde, 1, f[is-1+ldf*(je):], ldf)
					}
					//120          CONTINUE
				}
				//130       CONTINUE
			}
			if dscale != 0 {
				if (ijob == 1) || (ijob == 3) {
					dif = math.Sqrt(float64(2*m*n)) / (dscale * math.Sqrt(dsum))
				} else {
					dif = math.Sqrt(float64(pq)) / (dscale * math.Sqrt(dsum))
				}

				if (isolve == 2) && (iround == 1) {
					if notran {
						ifunc = ijob
					}
					scale2 = scale
					impl.Dlacpy(blas.All, m, n, c, ldc, work, m)
					impl.Dlacpy(blas.All, m, n, f, ldf, work[(m*n+1):], m)
					impl.Dlaset(blas.All, m, n, 0, 0, c, ldc)
					impl.Dlaset(blas.All, m, n, 0, 0, f, ldf)
				} else if (isolve == 2) && (iround == 2) {
					impl.Dlacpy(blas.All, m, n, work, m, c, ldc)
					impl.Dlacpy(blas.All, m, n, work[(m*n+1):], m, f, ldf)
					scale = scale2
				}
				//150    CONTINUE
			}
		}
	} else {
		//       Solve transposed (I, J)-subsystem
		//            A(I, I)**T * R(I, J)  + D(I, I)**T * L(I, J)  =  C(I, J)
		//            R(I, J)  * B(J, J)**T + L(I, J)  * E(J, J)**T = -F(I, J)
		//        for I = 1,2,..., P; J = Q, Q-1,..., 1
		scale = 1
		//DO 210 i = 1, p
		for i := 0; i <= p; i++ {
			is = iwork[i]
			ie = iwork[i+1] - 1
			mb = ie - is + 1
			//DO 200 j = q, p + 2, -1
			for j := q; j <= (p + 2); j-- {
				js = iwork[j]
				je = iwork[j+1] - 1
				nb = je - js + 1
				//dtgsy2( trans, ifunc, mb, nb, a( is, is ), lda,b( js, js ), ldb, c( is, js ), ldc,d( is, is ), ldd, e( js, js ), lde,f( is, js ), ldf, scaloc, dsum, dscale,iwork( q+2 ), ppqq, linfo )
				if linfo > 0 {
					info = linfo
				}
				if scaloc != 1 {
					//DO 160 k = 1, js - 1
					for k := 1; k <= (js - 1); k++ {
						bi.Dscal(m, scaloc, c[(ie+1)*ldc+k:], ldc)
						bi.Dscal(m, scaloc, f[(ie+1)*ldc+k:], ldf)
						// 160  CONTINUE
					}
					// DO 170 k = js, je
					for k := js; k <= je; k++ {
						bi.Dscal(is-1, scaloc, c[(ie+1)*ldc+k:], ldc)
						bi.Dscal(is-1, scaloc, f[(ie+1)*ldc+k:], ldf)
						//170  CONTINUE
					}
					//DO 180 k = js, je
					for k := js; k <= je; k++ {
						//CALL dscal( m-ie, scaloc, c( ie+1, k ), 1 )
						bi.Dscal(m-ie, scaloc, c[(ie+1)*ldc+k:], ldc)
						bi.Dscal(m-ie, scaloc, f[(ie+1)*ldc+k:], ldf)
						//180    CONTINUE
					}
					//DO 190 k = je + 1, n
					for k := je + 1; k <= n; k++ {
						bi.Dscal(m, scaloc, c[(ie+1)*ldc+k:], ldc)
						bi.Dscal(m, scaloc, f[(ie+1)*ldc+k:], ldf)
						//190   CONTINUE
					}
					scale = scale * scaloc
				}

				//              Substitute R(I, J) and L(I, J) into remaining equation.
				if j > (p + 2) {

					impl.Dgemm2('N', 'N', mb, js-1, nb, 1, c[js:], ldc, b[js:], ldb, 1, f[(is+ldf):], ldf)
					impl.Dgemm2('N', 'T', mb, js-1, nb, 1, f[js:], ldf, e[js:], lde, 1, f[is+ldf:], ldf) //que hago con la T?? a que tipo equivale?
				}
				if i < p {
					//CALL dgemm( 'T', 'N'         , m-ie, nb, mb, -one,a( is, ie+1 ), lda, c( is, js ), ldc, one,c( ie+1, js ), ldc )
					impl.Dgemm2('T', 'N', m-ie, nb, mb, -1, a[is-1+lda*(ie):], lda, c[is-1+ldc*(js-1):], ldc, 1, c[(ie)+ldc*(js-1):], ldc)
					//CALL dgemm( 'T', 'N',          m-ie, nb, mb, -one,d( is, ie+1 )     , ldd, f( is, js ), ldf, one,c( ie+1, js ), ldc )
					impl.Dgemm2('T', 'N', m-ie, nb, mb, -1, d[is-1+ldd*(ie):], ldd, f[is-1+ldf*(js-1):], ldf, 1, c[(ie)+ldc*(js-1):], ldc)

				}
			} // 200       CONTINUE
		} //210    CONTINUE
	}
	work[0] = float64(lwmin)
	return dif, scale
}
