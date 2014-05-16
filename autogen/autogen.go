package main

import (
	"bytes"
	"go/format"
	"log"
	"os"
	"path/filepath"
	"strings"
	"text/template"
)

type Unit struct {
	Name          string
	Receiver      string
	Offset        int    // From normal (for example, mass base unit is kg, not kg)
	PrintString   string // print string for the unit (kg for mass)
	ExtraConstant []Constant
	Suffix        string
	Singular      string
	TypeComment   string // Text to comment the type
	Dimensions    []Dimension
	ErForm        string //For Xxxer interface
}

type Dimension struct {
	Name  string
	Power int
}

const (
	TimeName   string = "TimeDim"
	LengthName string = "LengthDim"
	MassName   string = "MassDim"
)

type Constant struct {
	Name  string
	Value string
}

type Prefix struct {
	Name  string
	Power int
}

var Prefixes = []Prefix{
	{
		Name:  "Yotta",
		Power: 24,
	},
	{
		Name:  "Zetta",
		Power: 21,
	},
	{
		Name:  "Exa",
		Power: 18,
	},
	{
		Name:  "Peta",
		Power: 15,
	},
	{
		Name:  "Tera",
		Power: 12,
	},
	{
		Name:  "Giga",
		Power: 9,
	},
	{
		Name:  "Mega",
		Power: 6,
	},
	{
		Name:  "Kilo",
		Power: 3,
	},
	{
		Name:  "Hecto",
		Power: 2,
	},
	{
		Name:  "Deca",
		Power: 1,
	},
	{
		Name:  "",
		Power: 0,
	},
	{
		Name:  "Deci",
		Power: -1,
	},
	{
		Name:  "Centi",
		Power: -2,
	},
	{
		Name:  "Milli",
		Power: -3,
	},
	{
		Name:  "Micro",
		Power: -6,
	},
	{
		Name:  "Nano",
		Power: -9,
	},
	{
		Name:  "Pico",
		Power: -12,
	},
	{
		Name:  "Femto",
		Power: -15,
	},
	{
		Name:  "Atto",
		Power: -18,
	},
	{
		Name:  "Zepto",
		Power: -21,
	},
	{
		Name:  "Yocto",
		Power: -24,
	},
}

var Units = []Unit{
	{
		Name:        "Mass",
		Receiver:    "m",
		Offset:      -3,
		PrintString: "kg",
		Suffix:      "gram",
		Singular:    "Gram",
		TypeComment: "Mass represents a mass in kilograms",
		Dimensions: []Dimension{
			{
				Name:  MassName,
				Power: 1,
			},
		},
	},
	{
		Name:        "Length",
		Receiver:    "l",
		PrintString: "m",
		Suffix:      "meter",
		Singular:    "Meter",
		TypeComment: "Length represents a length in meters",
		Dimensions: []Dimension{
			{
				Name:  LengthName,
				Power: 1,
			},
		},
	},
	{
		Name:        "Time",
		Receiver:    "t",
		PrintString: "s",
		Suffix:      "second",
		Singular:    "Second",
		TypeComment: "Time represents a time in seconds",
		ExtraConstant: []Constant{
			{
				Name:  "Hour",
				Value: "3600",
			},
			{
				Name:  "Minute",
				Value: "60",
			},
		},
		Dimensions: []Dimension{
			{
				Name:  TimeName,
				Power: 1,
			},
		},
		ErForm: "Timer",
	},
}

var gopath string
var unitPkgPath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		log.Fatal("no gopath")
	}

	unitPkgPath = filepath.Join(gopath, "src", "github.com", "gonum", "unit")
}

// Generate generates a file for each of the units
func main() {
	for _, unit := range Units {
		generate(unit)
	}
}

const headerTemplate = `// This file is autogenerated by github.com/gonum/unit/autogen
// Changes should be made to the autogenerated template rather than this one

// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unit

import (
	"errors"
	"fmt"
	"math"
)

// {{.TypeComment}}
type {{.Name}} float64
`

const constTemplate = `
const(
	{{$unit := .Unit}}
	{{range $unit.ExtraConstant}} {{.Name}} {{$unit.Name}} = {{.Value}}
	{{end}}
	{{$prefixes := .Prefixes}}
	{{range $prefixes}} {{if .Name}} {{.Name}}{{$unit.Suffix}} {{else}} {{$unit.Singular}} {{end}} {{$unit.Name}} = {{if .Power}} 1e{{.Power}} {{else}} 1.0 {{end}}
	{{end}}
)
`

const methodTemplate = `
// Unit converts the {{.Name}} to a *Unit
func ({{.Receiver}} {{.Name}}) Unit() *Unit{
	return New(float64({{.Receiver}}), Dimensions{
		{{range .Dimensions}} {{.Name}}: {{.Power}},
		{{end}}
		})
}

// {{.Name}} allows {{.Name}} to implement a {{if .ErForm}}{{.ErForm}}{{else}}{{.Name}}er{{end}} interface
func ({{.Receiver}} {{.Name}}) {{.Name}}() {{.Name}} {
	return {{.Receiver}}
}

// From converts the unit into the receiver. From returns an
// error if there is a mismatch in dimension
func ({{.Receiver}} *{{.Name}}) From(u Uniter) error{
	if !DimensionsMatch(u, {{.Singular}}){
		*{{.Receiver}} = {{.Name}}(math.NaN())
		return errors.New("Dimension mismatch")
	}
	*{{.Receiver}} = {{.Name}}(u.Unit().Value())
	return nil
}
`

const formatTemplate = `
func ({{.Receiver}} {{.Name}}) Format(fs fmt.State, c rune){
	switch c {
	case 'v':
		if fs.Flag('#') {
			fmt.Fprintf(fs, "%T(%v)", {{.Receiver}}, float64({{.Receiver}}))
			return
		}
		fallthrough
	case 'e', 'E', 'f', 'F', 'g', 'G':
		p, pOk := fs.Precision()
		if !pOk {
			p = -1
		}
		w, wOk := fs.Width()
		if !wOk {
			w = -1
		}
		fmt.Fprintf(fs, "%*.*"+string(c), w, p, float64({{.Receiver}}))
		fmt.Fprint(fs, " {{.PrintString}}")
	default:
		fmt.Fprintf(fs, "%%!%c(%T=%g {{.PrintString}})", c, {{.Receiver}}, float64({{.Receiver}}))
	return
}
}
`

func generate(unit Unit) {
	lowerName := strings.ToLower(unit.Name)
	filename := filepath.Join(unitPkgPath, lowerName+".go")
	f, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Need to define new prefixes because text/template can't do math.
	// Need to do math because kilogram = 1 not 10^3

	prefixes := make([]Prefix, len(Prefixes))
	for i, p := range Prefixes {
		prefixes[i].Name = p.Name
		prefixes[i].Power = p.Power + unit.Offset
	}

	data := struct {
		Prefixes []Prefix
		Unit     Unit
	}{
		prefixes,
		unit,
	}

	buf := bytes.NewBuffer(make([]byte, 0))

	header := template.Must(template.New("header").Parse(headerTemplate))
	err = header.Execute(buf, unit)
	if err != nil {
		log.Fatal(err)
	}

	prefix := template.Must(template.New("prefix").Parse(constTemplate))
	err = prefix.Execute(buf, data)
	if err != nil {
		log.Fatal(err)
	}

	methods := template.Must(template.New("methods").Parse(methodTemplate))
	err = methods.Execute(buf, unit)
	if err != nil {
		log.Fatal(err)
	}

	form := template.Must(template.New("format").Parse(formatTemplate))
	err = form.Execute(buf, unit)
	if err != nil {
		log.Fatal(err)
	}

	b, err := format.Source(buf.Bytes())
	if err != nil {
		f.Write(buf.Bytes()) // This is here to debug bad format
		log.Fatalf("error formatting: %s", err)
	}

	f.Write(b)
}
