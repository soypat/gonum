package testlapack

import (
	"fmt"
	"testing"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/blas/blas64"
)

type Dtgexcer interface {
	Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int, q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, isWorkspaceQuery bool) (ifstOut, ilstOut int, illConditioned, lworkTooSmall bool)
}

func DtgexcTest(t *testing.T, impl Dtgexcer) {
	rnd := rand.New(rand.NewSource(1))
	const ldExtra = 5
	for _, n := range []int{0, 1, 2, 4, 8, 15, 16} {
		ldMin := max(1, n)
		for _, lda := range []int{ldMin, ldMin + ldExtra} {
			for _, ldb := range []int{ldMin, ldMin + ldExtra} {
				for _, ldq := range []int{ldMin, ldMin + ldExtra} {
					for _, ldz := range []int{ldMin, ldMin + ldExtra} {
						for _, wantq := range []bool{false, true} {
							for _, wantz := range []bool{false, true} {
								if n <= 2 {
									dtgexcTest(t, impl, rnd, wantq, wantz, n, lda, ldb, ldq, ldz, 0, 0)
									if n > 1 {
										dtgexcTest(t, impl, rnd, wantq, wantz, n, lda, ldb, ldq, ldz, 0, 1)
										dtgexcTest(t, impl, rnd, wantq, wantz, n, lda, ldb, ldq, ldz, 1, 0)
										dtgexcTest(t, impl, rnd, wantq, wantz, n, lda, ldb, ldq, ldz, 1, 1)
									}
									continue
								}
								maxIdx := max(0, n-1)
								for _, ifst := range []int{0, n / 2, maxIdx} {
									for _, ilst := range []int{0, n / 2, maxIdx, ifst / 2} {
										dtgexcTest(t, impl, rnd, wantq, wantz, n, lda, ldb, ldq, ldz, ifst, ilst)
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

func dtgexcTest(t *testing.T, impl Dtgexcer, rnd *rand.Rand, wantq, wantz bool, n, lda, ldb, ldq, ldz, ifst, ilst int) {
	const tol = 1e-14
	a, _, _ := randomSchurCanonical(n, lda, false, rnd)
	b, _, _ := randomSchurCanonical(n, ldb, false, rnd)
	// b := randomUpperTriGeneral(n, ldb, rnd)
	var q, z blas64.General
	if wantq {
		q = nanGeneral(n, n, ldq)
	}
	if wantz {
		z = nanGeneral(n, n, ldz)
	}
	// Compute the Schur form of the matrix pair (A, B).
	// The Schur form of A is stored in a.
	// The Schur form of B is stored in b.
	var query [1]float64
	_, _, qill, qbadWork := impl.Dtgexc(wantq, wantz, n, a.Data, a.Stride, b.Data, b.Stride, q.Data, ldq, z.Data, ldz, ifst, ilst, query[:], true)
	if qill || qbadWork {
		t.Fatal("bad return from workspace query")
	}
	lwork := int(query[0])
	work := make([]float64, lwork)
	ifstOut, ilstOut, ill, badWork := impl.Dtgexc(wantq, wantz, n, a.Data, a.Stride, b.Data, b.Stride, q.Data, ldq, z.Data, ldz, ifst, ilst, work, false)
	name := fmt.Sprintf("n=%d,lda=%d,ldb=%d,ifst=%d,ilst=%d", n, lda, ldb, ifst, ilst)
	if ill {
		return
		t.Fatalf("%s: ill conditioned matrix pair", name)
	}
	if badWork {
		t.Fatalf("%s: bad value of lwork", name)
	}
	if n == 0 {
		return // Nothing else to check.
	}
	if abs(ilstOut-ilst) > 1 || ilstOut >= n || ilstOut < 0 {
		t.Fatalf("%s: unexpected or OOB value of ilstOut: got:%v want:%v±1", name, ilstOut, ilst)
	}
	if ifstOut < 0 || ifstOut >= n {
		t.Fatalf("%s: OOB value of ifst: got:%v want:[0,%v)", name, ifst, n)
	}
}
