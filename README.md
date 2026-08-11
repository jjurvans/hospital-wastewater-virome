# Hospital wastewater reflects the virome of the patient population rather than community pathogen transmission

<img width="4303" height="1357" alt="README_monthly_viral_family_composition" src="https://github.com/user-attachments/assets/4dec011d-ba66-49e7-8557-3e01ad19a201" />


This repository contains the data and analysis scripts used to generate the main statistical analyses and figures for the manuscript **“Hospital wastewater reflects the virome of the patient population rather than community pathogen transmission”**.

The Python analyses can be run directly using GitHub Codespaces. The MaAsLin3 analysis used for Figure 5 was performed separately in R, and its output is provided so that Figure 5 can be reproduced without rerunning MaAsLin3.

## Repository contents

### Data

`EsVirutu_abundance_metadata.tsv`  
Main dataset containing viral abundance data and associated sample metadata used by the Python analysis scripts.

`maaslin3_species_abundance.tsv`  
Species-level abundance table used as input for the MaAsLin3 analysis.

`maaslin3_metadata.tsv`  
Sample metadata used as input for the MaAsLin3 analysis.

`Figure5_MaAsLin3_results.tsv`  
Combined MaAsLin3 results used as input for the Figure 5 Python script.

### Analysis scripts

`Figure2.py`  
Generates the analyses and visualisations associated with Figure 2.

`Figure3.py`  
Generates the analyses and visualisations associated with Figure 3 and Supplementary Figure S2, including temporal distance–decay, temporal stability, community-turnover analyses, and associated statistical tests.

`Figure4.py`  
Generates the analyses and visualisations associated with Figure 4, including rank–abundance, occupancy, prevalence, and species-specific occupancy analyses.

`Figure5_MaAsLin3_with_export.R`  
Performs the MaAsLin3 differential abundance and prevalence analyses for UH versus WWTP and UH versus pooled SA samples and exports the combined results as `Figure5_MaAsLin3_results.tsv`.

`Figure5.py`  
Generates the analyses and visualisations associated with Figure 5 using `EsVirutu_abundance_metadata.tsv` and the supplied `Figure5_MaAsLin3_results.tsv`.

## Wastewater environments and study locations

The following abbreviations are used throughout the data and scripts:

- **UH** – hospital wastewater
- **WWTP** – wastewater treatment plant influent
- **SA1 and SA2** – small sewer catchment areas
- **SA** – pooled SA1 and SA2 samples where indicated
- **TRE** – Tampere
- **KUO** – Kuopio

## Requirements

The Python analyses were performed using Python 3.11.9. Required Python packages and versions are provided in `requirements.txt`.

## Running the analyses in GitHub Codespaces

Open the repository in GitHub Codespaces. The development container installs the Python dependencies listed in `requirements.txt`.

Each analysis can then be run from the terminal:

```bash
python Figure2.py
python Figure3.py
python Figure4.py
python Figure5.py
```

Output figures and associated result tables are written directly to the repository working directory.

## Citation

If using these data or scripts, please cite:

**Hospital wastewater reflects the virome of the patient population rather than community pathogen transmission**

Full publication details will be added following publication.

---
### Contact

This repository is maintained by Jaana Jurvansuu, PhD, Adjunct Professor, Tampere University, AIvoa — Applied AI and Modelling for Biological Sciences.  

For collaboration or implementation enquiries, please connect via [LinkedIn](https://www.linkedin.com/in/jaanajurvansuu/).

### **Note**

The original scripts were generated with assistance from ChatGPT (OpenAI) and has been subsequently reviewed, modified, and tested by the repository author.

