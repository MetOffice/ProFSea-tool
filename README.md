# ProFSea <img width="100" alt="logo" src="https://github.com/user-attachments/assets/cba78630-9965-4b66-8be1-065ae975229e" />

[![DOI](https://zenodo.org/badge/713453340.svg)](https://zenodo.org/doi/10.5281/zenodo.10255467) [![Tests](https://github.com/MetOffice/ProFSea-tool/actions/workflows/pytest.yml/badge.svg?branch=profsea-v3)](https://github.com/MetOffice/ProFSea-tool/actions/workflows/pytest.yml) [![Lint](https://github.com/MetOffice/ProFSea-tool/actions/workflows/ruff.yml/badge.svg?branch=profsea-v3)](https://github.com/MetOffice/ProFSea-tool/actions/workflows/ruff.yml) [![Docs](https://github.com/MetOffice/ProFSea-tool/actions/workflows/deploy-docs.yml/badge.svg?branch=profsea-v3)](https://github.com/MetOffice/ProFSea-tool/actions/workflows/deploy-docs.yml) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

ProFSea is sea-level rise simulator based on statistical emulations of physical modelling experiments and lines of evidence from the IPCC and across the literature. It's modular, easy to setup and self-contained - all you need is global mean surface temperature and ocean heat content forcing anomalies, using any baseline period, and you're good to go.

We simulate sea-level change contributions from:

- Antartic Ice Sheet 🇦🇶
- Greenland Ice Sheet 🧊
- Glacier melt 🏔️
- Thermal expansion 🌡️
- Landwater ⛰️

ProFSea can calculate projections of both global mean sea-level rise and spatially-resolved sea-level change fields:

<p align="center">
  <img width="480" height="273" alt="output" src="https://github.com/user-attachments/assets/82122190-1566-4386-b558-da32b4bee7e7" />
</p>

All the data required to run the regionalisation module can be found here: [![DOI](https://doi.org/badge/10.5281/20427061.svg)](https://doi.org/10.5281/zenodo.20427061)

## Developments

Ongoing developments include:

- [x] 🛠️ Full spatial field projections of regional sea-level change
- [x] 🛠️ Use any input climate forcing to produce ProFSea projections
- [x] 🛠️ Updates to sea-level components based on the latest evidence in the literature
- [ ] 🛠️ Observational constraints to projection ensembles
- [x] 🛠️ Structural enhancements such as updated workflows, GitHub Actions and general reformatting
- [x] 🛠️ Full code documentation

## Contributors
Several people have contributed to the development of the ProFSea tool and User Guide documentation, namely: Rachel Perks, Jacob Cheung, Benjamin Harrison, Katie Hodge, Mathew Palmer, Michael Sanderson, Hamish Steptoe, Jennifer Weeks and Gregory Munday.

## Acknowledgements
This work was supported by the UK Research & Innovation (UKRI) Strategic Priorities Fund UK Climate Resilience programme. The programme is co-delivered by the Met Office and NERC on behalf of UKRI partners AHRC, EPSRC and ESRC. It was further supported by the Met Office Hadley Centre Climate Programme funded by BEIS and Defra.

## Licence
ProFSea is licensed under the [Open Government Licence 3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

If you have any queries about this tool please contact: enquiries@metoffice.gov.uk

<h5 align="center">
<img src="https://www.metoffice.gov.uk/binaries/content/gallery/metofficegovuk/images/about-us/website/mo_master_black_mono_for_light_backg_rbg.png" width="200" alt="Met Office"> <br>
&copy; British Crown Copyright 2023, Met Office <br> <br>
<a href="https://www.nationalarchives.gov.uk/doc/open-government-licence/"><img alt="Open Government Licence logo" src="https://www.nationalarchives.gov.uk/images/infoman/ogl-symbol-41px-retina-black.png"></a> 
</h5>
