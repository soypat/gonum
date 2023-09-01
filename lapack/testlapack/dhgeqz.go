// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/blas/blas64"
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
	rnd := rand.New(rand.NewSource(1))
	const ldaAdd = 5
	compvec := []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} // TODO: add lapack.SchurOrig
	for _, compq := range compvec {
		for _, compz := range compvec {
			for _, n := range []int{0, 1, 2, 3, 4, 20} {
				minLDA := max(1, n)
				for _, ldh := range []int{minLDA, n + ldaAdd} {
					for _, ldt := range []int{minLDA, n + ldaAdd} {
						for _, ldq := range []int{minLDA, n + ldaAdd} {
							for _, ldz := range []int{minLDA, n + ldaAdd} {
								for ilo := 0; ilo < n; ilo++ {
									for ihi := ilo + 1; ihi < n; ihi++ {
										testDhgeqz(t, rnd, impl, lapack.EigenvaluesOnly, compq, compz, n, ilo, ihi, ldh, ldt, ldq, ldz)
										testDhgeqz(t, rnd, impl, lapack.EigenvaluesAndSchur, compq, compz, n, ilo, ihi, ldh, ldt, ldq, ldz)
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
	generalFromComp := func(comp lapack.SchurComp, n, ld int, rnd *rand.Rand) blas64.General {
		switch comp {
		case lapack.SchurNone:
			return blas64.General{Stride: 1}
		case lapack.SchurHess:
			return nanGeneral(n, n, ld)
		case lapack.SchurOrig:
			panic("not implemented")
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

	// Query workspace needed.
	var query [1]float64
	impl.Dhgeqz(job, compq, compz, n, ilo, ihi, hg.Data, hg.Stride, tg.Data, tg.Stride, alphar, alphai, beta, q.Data, q.Stride, z.Data, z.Stride, query[:], true)
	lwork := int(query[0])
	if lwork < 1 {
		t.Fatal("bad lwork")
	}
	work := make([]float64, lwork)

	info := impl.Dhgeqz(job, compq, compz, n, ilo, ihi, hg.Data, hg.Stride, tg.Data, tg.Stride, alphar, alphai, beta, q.Data, q.Stride, z.Data, z.Stride, work, false)
	if info >= 0 {
		t.Error("got nonzero info", info)
	}

}
