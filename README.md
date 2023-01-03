
# *Explain-Da-V* - Explaining Dataset Changes for Semantic Data Versioning

***Explain-Da-V** is a framework aiming to explain changes between two given dataset versions. Explain-Da-V generates explanations that use data transformations to explain changes.*

<p align="center">
<img src ="/example_table_annotated.jpg">
</p>



## The Paper
[*Explaining Dataset Changes for Semantic Data Versioning with Explain-Da-V*](https://github.com/shraga89/ExplainDaV/blob/main/Explain_Da_V_TR.pdf)
Roee Shraga, Ren\'ee J. Miller, PVLDB (to appear), 2023

**BibTeX**:
TBD

## Getting Started

### Requirements
1. [Anaconda 3](https://www.anaconda.com/download/)
2. [Tabulate](https://pypi.org/project/tabulate/)
3. [Featuretools](https://www.featuretools.com/)

### Installation

 0. Download and extract the **[Semantic Data Versioning Benchmark](https://github.com/shraga89/ExplainDaV/tree/main/Semantic%20Data%20Versioning%20Benchmark%20\%28SDVB%29)**  
	0.1 For BYOD (bring your own data), please follow the format of SDVB
 1. Clone [Explain-Da-V repository](https://github.com/shraga89/ExplainDaV/tree/main/Explain-Da-V)  
	 1.1. Add three empty directories named `results` (stores functional dependencies), `temp`, and `output` (will hold the results of Explain-Da-V)  
	 1.2 Download [Metanome runnable](https://github.com/sekruse/metanome-cli/releases).  
	 1.3 Rename `metanome-cli-1.1.0.jar` as `metanome.jar` and add to the Explain-Da-V repository  
2. (optional) locate the dataset folder in the repository

### Running

 1. Configuring Explain-Da-V is done via the [`config`](https://github.com/shraga89/ExplainDaV/blob/main/Explain-Da-V/config.py) file  
	 1.1. (required) update the following entries to be consistent with local machine:  
		 - `dataset_name`: the name of the dataset (also the name of the folder, e.g., IMDB)  
		 - `problem_sets_file`: the location of the `problem_sets` file (e.g., 'Data/Benchmark/{}/problem_sets.csv'.format(dataset_name))  
	1.2 (optional) update other parameters, e.g., `CATEGORICAL_UPPER_BOUND`(the number of unique values to be considered as a categorical type).  
2. Run [`main`](https://github.com/shraga89/ExplainDaV/blob/main/Explain-Da-V/main.py)  
	2.1. Use `main_with_problem_sets` for default setting  
	2.2. Other settings are used for ablation study (e.g., `use_fd_discovery=False`) and baselines (e.g., `main_with_problem_sets_baseline_original(extend_for_auto_pipeline=False, extend_for_plus=True)`  
3. The output will be generated in the directory `output`  
	3.1 The output file documents the problem_set, nature of change (e.g., adding columns), the resolved trasformation and its evaluation (please see paper for more details).  
		  

### Acknowledgments
The code framework uses two existing systems, namely metanome and Foofah (both are included in the repository) :
* We find functional dependencies using [Metanome](https://hpi.de/naumann/projects/data-profiling-and-analytics/metanome-data-profiling.html). See [Functional Dependency Discovery: An Experimental Evaluation of Seven Algorithms](https://dl.acm.org/doi/pdf/10.14778/2794367.2794377) and [Repeatability - FDs and ODs](https://hpi.de/naumann/projects/repeatability/data-profiling/fds.html) for additional details.
* We adopt and extend [Foofah](https://github.com/umich-dbgroup/foofah) to resolve textual transformations. See [Foofah: Transforming Data By Example](https://dl.acm.org/doi/pdf/10.1145/3035918.3064034) for additional details.


## The Team
*Explain-Da-V* was developed at the [Data Lab](https://db.khoury.northeastern.edu/), [Northeastern University](https://www.northeastern.edu/) by [Dr. Roee Shraga](https://sites.google.com/view/roee-shraga/) and [Prof. Ren\'ee J. Miller](https://www.khoury.northeastern.edu/people/renee-miller/).

**The repository also contains Ablation Study Plots (Figure 11, Section 7.3):**
* [Validity](https://github.com/shraga89/ExplainDaV/blob/main/Figures/validity_ablation.pdf)
* [Generalizability](https://github.com/shraga89/ExplainDaV/blob/main/Figures/generalizability_ablation.pdf)
* [Conciseness](https://github.com/shraga89/ExplainDaV/blob/main/Figures/conciseness_ablation.pdf)
* [Concentration](https://github.com/shraga89/ExplainDaV/blob/main/Figures/concentration_ablation.pdf)

