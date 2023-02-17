---
title: 'Waveplot: An online wavefunction viewer'
tags:
  - Python
  - Chemistry
  - Computational chemistry
  - Quantum chemistry
  - Single molecule magnets
authors:
  - name: Jon G. C. Kragskow
    orcid: 0000-0001-9488-8204
    affiliation: 1
affiliations:
 - name: Department of Chemistry, University of Bath, Bath, BA2 7AY, United Kingdom
   index: 1
date: 17 February 2023
bibliography: paper.bib
---

# Summary
The Schrodinger equation is routinely employed throughout physical chemistry to describe
a variety of chemical and physical processes. In particular, Hydrogenic orbitals are universally used
for visualisation of electron density as a function of position within a molecule or atom.
While these are merely approximate for molecules and atoms other than hydrogen, their usage is widespread
and takes the form of either three-dimensional isosurfaces describing both angular and radial behaviour,
or two-dimensional representations describing radial behaviour alone. The quantum harmonic oscillator is another
example of the Schrodinger equation's application to chemistry, as it is routinely employed in the description of 
molecular vibrational modes which are encountered in a variety of measurements and fields such as infrared and Raman
spectroscopies, chemical kinetics, and luminescent materials.


# Statement of need

`waveplot` is an online wavefunction viewer for hydrogenic orbitals, harmonic oscillators,
and 4f electron densities written in Python using Dash.[] Python allows for the
both the computation of the raw data required to descibe these phenomena, and
for the construction of a user-friendly web interface through the use of Dash.
Waveplot makes all of its raw data available in plain text format, allowing the end user
to create their own figures, alongside those provided by the website. `waveplot` was designed
to be used by both chemistry researchers to produce publication and presentation quality data
and by students in core chemistry courses to supplement their understanding through the use
of interactive demonstrations.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

I thank Dr. Daniel Reta for insightful discussions on this project.

# References
