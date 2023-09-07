// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"testing"
	"unsafe"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/lapack"
)

/*
#cgo CFLAGS: -I/home/pato/src/ongoing/lapack/SRC/  -I/home/pato/src/ongoing/lapack/BLAS/SRC
#cgo LDFLAGS: -L/home/pato/src/ongoing/lapack -llapack -lrefblas -lgfortran -lm -ltmglib


void dhgeqz_(char * JOB, char * COMPQ, char * COMPZ, int * N, int * ILO, int * IHI, double * H, int * LDH, double * T, int * LDT, double * ALPHAR, double * ALPHAI, double * BETA, double * Q, int * LDQ, double * Z, int * LDZ, double * WORK, int * LWORK, int * INFO);
*/
import "C"

type Dhgeqzer interface {
	Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int,
		h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta,
		q []float64, ldq int, z []float64, ldz int, work []float64, workspaceQuery bool) (info int)
}

func DhgeqzTest(t *testing.T, impl Dhgeqzer) {
	src := uint64(8878)
	rnd := rand.New(rand.NewSource(src))
	const ldaAdd = 0
	compvec := []lapack.SchurComp{lapack.SchurNone}
	for _, compq := range compvec {
		for _, compz := range compvec {
			for _, n := range []int{2, 3, 4, 16} {
				minLDA := max(1, n)
				for _, ldh := range []int{minLDA, n + ldaAdd} {
					for _, ldt := range []int{minLDA, n + ldaAdd} {
						for _, ldq := range []int{minLDA, n + ldaAdd} {
							for _, ldz := range []int{minLDA, n + ldaAdd} {
								for ilo := 0; ilo < n; ilo++ {
									for ihi := ilo; ihi < n; ihi++ {
										for cas := 0; cas < 1000; cas++ {
											testDhgeqz(t, rnd, impl, lapack.EigenvaluesAndSchur, compq, compz, n, ilo, ihi, ldh, ldt, ldq, ldz)
											testDhgeqz(t, rnd, impl, lapack.EigenvaluesOnly, compq, compz, n, ilo, ihi, ldh, ldt, ldq, ldz)
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

func testDhgeqz(t *testing.T, rnd *rand.Rand, impl Dhgeqzer, job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi, ldh, ldt, ldq, ldz int) {
	tol := 1e-12
	if n > 10 {
		tol = 5e-2
	}
	name := fmt.Sprintf("Case job=%q, compq=%q, compz=%q, n=%v, ilo=%v, ihi=%v, ldh=%v, ldt=%v, ldq=%v, ldz=%v",
		job, compq, compz, n, ilo, ihi, ldh, ldt, ldq, ldz)
	generalFromComp := func(comp lapack.SchurComp, n, ld int, rnd *rand.Rand) blas64.General {
		switch comp {
		case lapack.SchurNone:
			return blas64.General{Stride: 1}
		case lapack.SchurHess:
			return nanGeneral(n, n, ld)
		case lapack.SchurOrig:
			return randomOrthogonal(n, rnd)
		default:
			panic("bad comp")
		}
	}
	hg := randomHessenberg(n, ldh, rnd)
	tg := upperTriGeneral(n, n, ldt, rnd)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	q := generalFromComp(compq, n, ldq, rnd)
	z := generalFromComp(compz, n, ldz, rnd)
	hCopy := cloneGeneral(hg)
	tCopy := cloneGeneral(tg)
	alpharWant := append([]float64{}, alphar...)
	alphaiWant := append([]float64{}, alphai...)
	betaWant := append([]float64{}, beta...)
	tgWant := cloneGeneral(tg)
	hgWant := cloneGeneral(hg)
	qWant := cloneGeneral(q)
	zWant := cloneGeneral(z)
	var qCopy, zCopy blas64.General
	if compq != lapack.SchurNone {
		qCopy = cloneGeneral(q)
	}
	if compz != lapack.SchurNone {
		zCopy = cloneGeneral(z)
	}

	// Query workspace needed.
	var query [1]float64
	impl.Dhgeqz(job, compq, compz, n, ilo, ihi, hg.Data, hg.Stride, tg.Data, tg.Stride, alphar, alphai, beta, q.Data, q.Stride, z.Data, z.Stride, query[:], true)
	lwork := int(query[0])
	if lwork < 1 {
		t.Fatal("bad lwork")
	}

	work := make([]float64, lwork)
	workWant := append([]float64{}, work...)
	info := impl.Dhgeqz(job, compq, compz, n, ilo, ihi, hg.Data, hg.Stride, tg.Data, tg.Stride, alphar, alphai, beta, q.Data, q.Stride, z.Data, z.Stride, work, false)
	if info >= 0 {
		t.Error("got nonzero info", info)
	}

	if job == lapack.EigenvaluesAndSchur {
		// if !isSchurCanonicalGeneral(hg, tol) {
		// 	t.Error("H not in Schur canonical form", hg)
		// }

	}

	// Compute right and left hand sides of
	//   H*x = lambda*T*x
	// Where lambda are the eigenvalues computed by Dhgeqz.
	// lhs := blas64.Vector{N: n, Data: make([]float64, n), Inc: 1}
	// rhs := blas64.Vector{N: n, Data: make([]float64, n), Inc: 1}
	// x := blas64.Vector{
	// 	N:    n,
	// 	Data: randomSlice(n, rnd),
	// 	Inc:  1,
	// }
	// blas64.Gemv(blas.NoTrans, 1, hCopy, x, 0, lhs)
	// for i := 0; i < n; i++ {
	// 	if alphar[i] == 0 && alphai[i] == 0 && beta[i] == 0 {
	// 		continue // singular EV.
	// 	}
	// 	lambdar := alphar[i]
	// 	lambdai := alphai[i]
	// 	if beta[i] != 0 {
	// 		lambdai /= beta[i]
	// 		lambdar /= beta[i]
	// 	}
	// 	if lambdai != 0 {
	// 		continue // Imag EV.
	// 	}

	// 	blas64.Gemv(blas.NoTrans, lambdar, tCopy, x, 0, rhs)
	// 	if !floats.EqualApprox(lhs.Data, rhs.Data, tol) {
	// 		t.Error("H*x != lambda*T*x", i, complex(lambdar, 0), lhs.Data, rhs.Data)
	// 	}
	// }
	// if t.Failed() {
	// 	t.Fatal("exit")
	// }

	// fmt.Println(alphar, alphai, beta)

	infoWant := _lapack{}.Dhgeqz(job, compq, compz, n, ilo, ihi, hgWant.Data, hgWant.Stride, tgWant.Data, tgWant.Stride, alpharWant, alphaiWant, betaWant, qWant.Data, qWant.Stride, zWant.Data, zWant.Stride, workWant, false)
	if info != infoWant {
		t.Errorf("info mismatch: got %v, want %v", info, infoWant)
	}

	// if !equalApproxGeneral(hg, hgWant, tol) {
	// 	hc := cloneGeneral(hg)
	// 	floats.Sub(hc.Data, hgWant.Data)
	// 	t.Error("H not equal\nmaxdif=", floats.Max(hc.Data), "\n", hg, "\n", hgWant)
	// }
	// if !equalApproxGeneral(tg, tgWant, tol) {
	// 	_, _ = qCopy, zCopy
	// 	tc := cloneGeneral(tCopy)
	// 	floats.Sub(tc.Data, tg.Data)
	// 	t.Error("T not equal\nmaxdif=", floats.Max(tc.Data), "\n", tg, "\n", tgWant)
	// }
	if !equalApproxGeneral(q, qWant, tol) {
		qc := cloneGeneral(q)
		floats.Sub(qc.Data, qWant.Data)
		t.Error("Q not equal\nmaxdif=", floats.Max(qc.Data), "\n", q, "\n", qWant)
	}
	if !equalApproxGeneral(z, zWant, tol) {
		zc := cloneGeneral(z)
		floats.Sub(zc.Data, zWant.Data)
		t.Error("Q not equal\nmaxdif=", floats.Max(zc.Data), "\n", q, "\n", qWant)
	}
	if !floats.EqualApprox(alphar, alpharWant, tol) {
		t.Error("alphar not equal", alphar, alpharWant)
	}
	if !floats.EqualApprox(alphai, alphaiWant, tol) {
		t.Error("alphai not equal", alphai, alphaiWant)
	}
	if !floats.EqualApprox(beta, betaWant, tol) {
		t.Error("beta not equal", beta, betaWant)
	}
	if t.Failed() {
		t.Errorf("%#v", hCopy.Data)
		t.Errorf("%#v", tCopy.Data)
		printFortranReshape("h", hCopy.Data, true, true, n, ldh)
		printFortranReshape("t", tCopy.Data, true, true, n, ldt)
		work = make([]float64, len(work))
		alphai = make([]float64, len(alphai))
		alphar = make([]float64, len(alphar))
		beta = make([]float64, len(beta))
		// impl.Dhgeqz(job, compq, compz, n, ilo, ihi, hCopy.Data, hCopy.Stride, tCopy.Data, tCopy.Stride, alphar, alphai, beta, qCopy.Data, ldq, zCopy.Data, ldz, work, false)
		t.Fatal(name, "exit")
	}

	_ = qCopy
	_ = zCopy
}

type _lapack struct{}

func (_lapack) Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int,
	h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta,
	q []float64, ldq int, z []float64, ldz int, work []float64, workspaceQuery bool) (info int) {
	punf := pun[float64, C.double]
	puni := pun[int, C.int]
	cjob := C.char(job)
	ccompq := C.char(compq)
	ccompz := C.char(compz)
	lwork := len(work)
	if workspaceQuery {
		lwork = -1
	}
	var qptr, zptr *float64
	if compq != lapack.SchurNone {
		qptr = &q[0]
		defer transposeCurry(n, q, ldq)()
	}
	if compz != lapack.SchurNone {
		zptr = &z[0]
		defer transposeCurry(n, z, ldz)()
	}
	// convert to Fortran indexing.
	ilo++
	ihi++
	defer transposeCurry(n, h, ldh)()
	defer transposeCurry(n, t, ldt)()

	C.dhgeqz_(&cjob, &ccompq, &ccompz, puni(&n), puni(&ilo), puni(&ihi), punf(&h[0]), puni(&ldh), punf(&t[0]), puni(&ldt), punf(&alphar[0]), punf(&alphai[0]), punf(&beta[0]), punf(qptr), puni(&ldq), punf(zptr), puni(&ldz), punf(&work[0]), puni(&lwork), puni(&info))
	return info - 1
}

func transposeInPlace(n int, a []float64, lda int) {
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			a[i*lda+j], a[j*lda+i] = a[j*lda+i], a[i*lda+j]
		}
	}
}

func transposeCurry(n int, a []float64, lda int) func() {
	transposeInPlace(n, a, lda)
	return func() {
		transposeInPlace(n, a, lda)
	}
}

func pun[F, T any](p *F) *T {
	return (*T)(unsafe.Pointer(p))
}
