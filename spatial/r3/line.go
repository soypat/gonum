package r3

// Line defines an infinite line in 3D space.
type Line struct {
	// Point on line.
	p Vec
	// Direction of line in space. Is unitary.
	n Vec
}

func NewLine(p, n Vec) Line {
	return Line{
		p: p,
		n: Unit(n),
	}
}

// Distance returns the minimum euclidean distance of point p to l.
func (l Line) Distance(p Vec) float64 {
	// https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
	pOnLine := Add(l.p, l.n)
	num := Norm(Cross(Sub(p, l.p), Sub(p, pOnLine)))
	return num / Norm(Sub(pOnLine, l.p))
}

// Closest returns the closest point on l to p.
func (l Line) Closest(p Vec) Vec {
	pOnLine := Add(l.p, l.n)
	// https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
	t := -Dot(Sub(l.p, p), Sub(pOnLine, p)) / Norm2(Sub(pOnLine, l.p))
	return l.eval(t)
}

// eval returns the point on l at a distance t from the initial point
// in the lines direction.
func (l Line) eval(t float64) Vec {
	return Add(l.p, Scale(t, l.n))
}
